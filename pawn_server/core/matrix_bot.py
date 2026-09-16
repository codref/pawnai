"""Inbound Matrix chatbot worker for ``pawn-server serve``.

Talks to Matrix via matrix-nio and runs agent turns in-process
(``run_agent_turn``) — same tools/skills/memory as ``pawn-agent chat``.
Does not call the HTTP API.

Install: ``uv sync --extra matrix``
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Matrix event content soft limit; leave headroom under the hard ~65k byte cap.
_MAX_CHUNK = 30_000
# Long-poll sync; keep under typical reverse-proxy idle timeouts when possible.
_SYNC_TIMEOUT_MS = 30_000
_SYNC_BACKOFF_START_S = 1.0
_SYNC_BACKOFF_MAX_S = 60.0


# ── Pure helpers (unit-tested without nio) ────────────────────────────────────


def conversation_id(room_id: str) -> str:
    """Stable sallm conversation key for a Matrix room (not a diarization id)."""
    return f"matrix:{room_id}"


def is_direct_room(member_count: int) -> bool:
    """Heuristic: 1–2 members ≈ DM; larger rooms need the command prefix."""
    return member_count <= 2


def normalize_body(body: str, *, is_edit: bool = False, new_body: Optional[str] = None) -> str:
    """Prefer edit payload; drop Matrix reply quote fallbacks."""
    text = (new_body if is_edit and new_body else body) or ""
    if is_edit and text.startswith("* "):
        text = text[2:]
    # Reply fallbacks look like "> <@user:hs> line\\n\\nactual message"
    if text.startswith(">"):
        parts = text.split("\n\n", 1)
        if len(parts) == 2:
            text = parts[1]
    return text.strip()


def extract_prompt(
    body: str,
    *,
    command_prefix: str,
    is_dm: bool,
) -> Optional[str]:
    """Return agent prompt text, or None if this room message should be ignored."""
    text = body.strip()
    if not text:
        return None
    if text.startswith(command_prefix):
        return text[len(command_prefix) :].lstrip() or None
    if is_dm:
        return text
    return None


def chunk_text(text: str, limit: int = _MAX_CHUNK) -> list[str]:
    """Split long replies so room_send stays under Matrix size limits."""
    if len(text) <= limit:
        return [text]
    return [text[i : i + limit] for i in range(0, len(text), limit)]


def matrix_reply_body(raw: str) -> str:
    """Drop CLI ``[tool]`` trail lines; Matrix users only need the answer."""
    text = (raw or "").strip()
    if not text.startswith("[tool]"):
        return text
    # handle_user_input joins tool lines + "\\n\\n" + answer
    parts = text.split("\n\n", 1)
    if len(parts) == 2 and parts[1].strip():
        return parts[1].strip()
    # No answer body — strip tool prefixes line by line
    kept = [ln for ln in text.splitlines() if not ln.startswith("[tool]")]
    return "\n".join(kept).strip() or text


def verification_allowed(
    sender: str,
    *,
    bot_user_id: str,
    inviters: list[str],
) -> bool:
    """Own-account cross-device verify always; else allowlist (empty = open)."""
    if sender == bot_user_id:
        return True
    if not inviters:
        return True
    return sender in inviters


def _require_nio():
    try:
        import nio  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "matrix-nio is required for matrix_bot. Install with: uv sync --extra matrix"
        ) from exc


def _patch_nio_sas_for_element() -> None:
    """Fix matrix-nio 0.26 SAS interop with Element.

    1. Commitment must be unpadded base64, not hex (nio#570).
    2. Prefer ``hkdf-hmac-sha256.v2``; nio advertises v1 but uses the v2
       calculator, so Element rejects MACs as key mismatch.
    """
    import base64
    from hashlib import sha256

    from nio.api import Api
    from nio.crypto.sas import Sas, SasState
    from nio.event_builders import ToDeviceMessage

    if getattr(Sas, "_pawn_sas_patched", False):
        return

    _MAC_V1 = "hkdf-hmac-sha256"
    _MAC_V2 = "hkdf-hmac-sha256.v2"

    def _unpadded_b64(data: bytes) -> str:
        return base64.b64encode(data).decode("ascii").rstrip("=")

    def _mac_fn(self):
        assert self.established_sas
        # v2 / fixed encoding; legacy v1 name needs the broken base64 variant.
        if self.chosen_mac_method == _MAC_V1:
            return self.established_sas.calculate_mac_invalid_base64
        return self.established_sas.calculate_mac

    _orig_from_start = Sas.from_key_verification_start

    @classmethod  # type: ignore[misc]
    def _from_start(cls, own_user, own_device, own_fp_key, other_olm_device, event):
        obj = _orig_from_start.__func__(
            cls, own_user, own_device, own_fp_key, other_olm_device, event
        )
        string_content = Api.to_canonical_json(event.source["content"])
        obj.commitment = _unpadded_b64(
            sha256(obj.pubkey.encode() + string_content.encode()).digest()
        )
        macs = list(event.message_authentication_codes or [])
        # Orig cancels if v1 absent; revive when Element only offers v2.
        if obj.state == SasState.canceled and _MAC_V2 in macs:
            obj.state = SasState.started
            obj.cancel_code = None
            obj.cancel_reason = None
        return obj

    def _check_commitment(self, key: str) -> bool:
        assert self.commitment
        calculated = _unpadded_b64(
            sha256(
                key.encode()
                + Api.to_canonical_json(self.start_verification().content).encode()
            ).digest()
        )
        return self.commitment == calculated

    def _accept_verification(self) -> ToDeviceMessage:
        if self.we_started_it:
            from nio.exceptions import LocalProtocolError

            raise LocalProtocolError(
                "Verification was started by us, can't accept offer."
            )
        if self.state == SasState.canceled:
            from nio.exceptions import LocalProtocolError

            raise LocalProtocolError(
                "SAS verification was canceled, can't accept offer."
            )

        sas_methods = []
        if "emoji" in self.short_auth_string:
            sas_methods.append("emoji")
        if "decimal" in self.short_auth_string:
            sas_methods.append("decimal")

        macs = list(self.mac_methods or [])
        self.chosen_mac_method = _MAC_V2 if _MAC_V2 in macs else _MAC_V1

        if Sas._key_agreement_v2 in self.key_agreement_protocols:
            self.chosen_key_agreement = Sas._key_agreement_v2
        else:
            self.chosen_key_agreement = Sas._key_agreement_v1

        content = {
            "transaction_id": self.transaction_id,
            "key_agreement_protocol": self.chosen_key_agreement,
            "hash": self._hash_v1,
            "message_authentication_code": self.chosen_mac_method,
            "short_authentication_string": sas_methods,
            "commitment": self.commitment,
        }
        return ToDeviceMessage(
            "m.key.verification.accept",
            self.other_olm_device.user_id,
            self.other_olm_device.id,
            content,
        )

    def _get_mac(self) -> ToDeviceMessage:
        if not self.sas_accepted:
            from nio.exceptions import LocalProtocolError

            raise LocalProtocolError("SAS string wasn't yet accepted")
        if self.state == SasState.canceled:
            from nio.exceptions import LocalProtocolError

            raise LocalProtocolError(
                "SAS verification was canceled, can't generate MAC."
            )

        key_id = f"ed25519:{self.own_device}"
        calculate_mac = _mac_fn(self)
        info = (
            "MATRIX_KEY_VERIFICATION_MAC"
            f"{self.own_user}{self.own_device}"
            f"{self.other_olm_device.user_id}{self.other_olm_device.id}"
            f"{self.transaction_id}"
        )
        mac = {key_id: calculate_mac(self.own_fp_key, info + key_id)}
        content = {
            "mac": mac,
            "keys": calculate_mac(key_id, info + "KEY_IDS"),
            "transaction_id": self.transaction_id,
        }
        return ToDeviceMessage(
            "m.key.verification.mac",
            self.other_olm_device.user_id,
            self.other_olm_device.id,
            content,
        )

    def _receive_mac_event(self, event) -> None:
        """Like nio's handler, but use the MAC fn matching chosen_mac_method."""
        if self.verified:
            return
        if not self._event_ok(event):
            return
        if self.state != SasState.key_received:
            self.state = SasState.canceled
            self.cancel_code, self.cancel_reason = Sas._unexpected_message_error
            return

        info = (
            f"MATRIX_KEY_VERIFICATION_MAC{self.other_olm_device.user_id}"
            f"{self.other_olm_device.id}{self.own_user}{self.own_device}"
            f"{self.transaction_id}"
        )
        key_ids = ",".join(sorted(event.mac.keys()))
        calculate_mac = _mac_fn(self)

        if event.keys != calculate_mac(key_ids, info + "KEY_IDS"):
            # Element may include cross-signing keys; still try device key alone.
            logger.warning(
                "SAS KEY_IDS MAC mismatch (will still check device key); "
                "chosen_mac_method=%s",
                self.chosen_mac_method,
            )

        for key_id, key_mac in event.mac.items():
            try:
                key_type, device_id = key_id.split(":", 1)
            except ValueError:
                continue
            if key_type != "ed25519" or device_id != self.other_olm_device.id:
                continue
            other_fp_key = self.other_olm_device.ed25519
            if key_mac != calculate_mac(other_fp_key, info + key_id):
                self.state = SasState.canceled
                self.cancel_code, self.cancel_reason = self._key_mismatch_error
                return
            self.verified_devices.append(device_id)

        if not self.verified_devices:
            self.state = SasState.canceled
            self.cancel_code, self.cancel_reason = self._key_mismatch_error
            return
        self.state = SasState.mac_received

    Sas.from_key_verification_start = _from_start  # type: ignore[method-assign]
    Sas._check_commitment = _check_commitment  # type: ignore[method-assign]
    Sas.accept_verification = _accept_verification  # type: ignore[method-assign]
    Sas.get_mac = _get_mac  # type: ignore[method-assign]
    Sas.receive_mac_event = _receive_mac_event  # type: ignore[method-assign]
    Sas._pawn_sas_patched = True  # type: ignore[attr-defined]
    logger.info("Patched matrix-nio SAS for Element (commitment + MAC v2)")


def _validate_cfg(mb: Any) -> None:
    if not mb.homeserver_url or not mb.user_id:
        raise RuntimeError("matrix_bot.homeserver_url and matrix_bot.user_id are required")
    if not mb.user_token and not mb.user_password:
        raise RuntimeError("matrix_bot needs user_token or user_password")


# ── Client lifecycle ──────────────────────────────────────────────────────────


def _build_client(mb: Any):
    from nio import AsyncClient, AsyncClientConfig

    # Reuse device_id + store_path across restarts; a fresh device_id in E2EE
    # rooms often means silent message drops until re-verified.
    Path(mb.store_path).mkdir(parents=True, exist_ok=True)
    return AsyncClient(
        mb.homeserver_url,
        mb.user_id,
        device_id=mb.device_id,
        store_path=mb.store_path,
        config=AsyncClientConfig(
            max_limit_exceeded=0,
            max_timeouts=0,
            store_sync_tokens=True,
            encryption_enabled=True,
        ),
    )


async def _login(client: Any, mb: Any) -> None:
    from nio import LoginError

    if mb.user_token:
        client.access_token = mb.user_token
        client.user_id = mb.user_id
        client.load_store()
        if client.should_upload_keys:
            await client.keys_upload()
        return

    resp = await client.login(password=mb.user_password, device_name=mb.device_name)
    if isinstance(resp, LoginError):
        raise RuntimeError(f"Matrix login failed: {resp.message}")


async def _send_text(client: Any, room_id: str, text: str) -> None:
    # Clients render formatted_body as HTML; body stays plain markdown fallback.
    from markdown import markdown

    body = matrix_reply_body(text)
    for chunk in chunk_text(body):
        content = {
            "msgtype": "m.text",
            "body": chunk,
            "format": "org.matrix.custom.html",
            "formatted_body": markdown(
                chunk,
                extensions=["fenced_code", "nl2br", "sane_lists"],
            ),
        }
        await client.room_send(
            room_id,
            "m.room.message",
            content,
            ignore_unverified_devices=True,
        )


# ── Event handlers ────────────────────────────────────────────────────────────


def _register_callbacks(client: Any, cfg: Any, registry: Any) -> None:
    from nio import (
        InviteMemberEvent,
        KeyVerificationCancel,
        KeyVerificationEvent,
        KeyVerificationKey,
        KeyVerificationMac,
        KeyVerificationStart,
        LocalProtocolError,
        MegolmEvent,
        RoomMessageText,
        ToDeviceError,
        ToDeviceMessage,
        UnknownToDeviceEvent,
    )

    mb = cfg.matrix_bot

    def _allowed(sender: str) -> bool:
        return verification_allowed(
            sender,
            bot_user_id=client.user_id,
            inviters=list(mb.inviters or []),
        )

    async def on_to_device(event: Any) -> None:
        # Modern Element: request → ready → start → key → mac → done.
        # nio often fails Element's cross-signing MACs ("expected key did not
        # match"); we still verify_device() after emoji confirm so trust sticks.
        try:
            etype = (getattr(event, "source", {}) or {}).get("type", "")

            if etype == "m.key.verification.request":
                content = event.source.get("content") or {}
                if not _allowed(event.sender):
                    logger.warning("Ignoring verification request from %s", event.sender)
                    return
                if "m.sas.v1" not in (content.get("methods") or []):
                    logger.warning("Verification request without SAS from %s", event.sender)
                    return
                txid = content["transaction_id"]
                logger.info("Verification request from %s — sending ready", event.sender)
                ready = ToDeviceMessage(
                    type="m.key.verification.ready",
                    recipient=event.sender,
                    recipient_device=content["from_device"],
                    content={
                        "from_device": client.device_id,
                        "methods": ["m.sas.v1"],
                        "transaction_id": txid,
                    },
                )
                resp = await client.to_device(ready, txid)
                if isinstance(resp, ToDeviceError):
                    logger.error("verification ready failed: %s", resp)

            elif isinstance(event, KeyVerificationStart):
                if not _allowed(event.sender):
                    logger.warning("Ignoring verification start from %s", event.sender)
                    return
                if "emoji" not in (event.short_authentication_string or []):
                    logger.warning(
                        "Verification without emoji from %s: %s",
                        event.sender,
                        event.short_authentication_string,
                    )
                    return
                resp = await client.accept_key_verification(event.transaction_id)
                if isinstance(resp, ToDeviceError):
                    logger.error("accept_key_verification failed: %s", resp)
                    return
                sas = client.key_verifications[event.transaction_id]
                resp = await client.to_device(sas.share_key())
                if isinstance(resp, ToDeviceError):
                    logger.error("share_key failed: %s", resp)

            elif isinstance(event, KeyVerificationKey):
                sas = client.key_verifications[event.transaction_id]
                logger.info(
                    "SAS emojis: %s — click They match in Element",
                    sas.get_emoji(),
                )
                # Send our MAC now (Element verifies it after you confirm).
                resp = await client.confirm_short_auth_string(event.transaction_id)
                if isinstance(resp, ToDeviceError):
                    logger.error("confirm_short_auth_string failed: %s", resp)
                    return
                other = getattr(sas, "other_olm_device", None)
                if other is not None:
                    client.verify_device(other)
                    logger.info(
                        "Marked device verified: %s %s (mac_method=%s)",
                        other.user_id,
                        other.device_id,
                        getattr(sas, "chosen_mac_method", "?"),
                    )

            elif isinstance(event, KeyVerificationMac):
                sas = client.key_verifications.get(event.transaction_id)
                if sas is None or sas.canceled:
                    logger.warning(
                        "Ignoring MAC for canceled/unknown verification %s",
                        getattr(event, "transaction_id", "?"),
                    )
                    return
                if not sas.sas_accepted:
                    sas.accept_sas()
                try:
                    mac_msg = sas.get_mac()
                except LocalProtocolError as exc:
                    logger.warning("Verification MAC skipped: %s", exc)
                    return
                resp = await client.to_device(mac_msg)
                if isinstance(resp, ToDeviceError):
                    logger.error("verification MAC send failed: %s", resp)
                    return
                other = getattr(sas, "other_olm_device", None)
                if other is not None:
                    client.verify_device(other)
                    done = ToDeviceMessage(
                        type="m.key.verification.done",
                        recipient=event.sender,
                        recipient_device=other.device_id,
                        content={"transaction_id": sas.transaction_id},
                    )
                    resp = await client.to_device(done, sas.transaction_id)
                    if isinstance(resp, ToDeviceError):
                        logger.error("verification done failed: %s", resp)
                logger.info(
                    "Device verification MAC exchanged with %s (verified=%s)",
                    event.sender,
                    sas.verified,
                )

            elif etype == "m.key.verification.done":
                txid = (event.source.get("content") or {}).get("transaction_id")
                sas = client.key_verifications.get(txid) if txid else None
                logger.info(
                    "Verification finished with %s (verified=%s devices=%s)",
                    event.sender,
                    getattr(sas, "verified", None),
                    getattr(sas, "verified_devices", None),
                )

            elif isinstance(event, KeyVerificationCancel):
                # Element often cancels after emoji with this reason even when UI
                # goes green; device was already trusted on KeyVerificationKey.
                logger.info(
                    "Verification cancelled by %s: %s "
                    "(ok if device was already marked verified above)",
                    event.sender,
                    getattr(event, "reason", ""),
                )
        except Exception:
            logger.exception("Key verification handler failed")

    async def on_message(room: Any, event: Any) -> None:
        if event.sender == client.user_id:
            return

        source = getattr(event, "source", {}) or {}
        content = source.get("content", {}) if isinstance(source, dict) else {}
        relates = content.get("m.relates_to") or {}
        is_edit = relates.get("rel_type") == "m.replace"
        new_body = (content.get("m.new_content") or {}).get("body")

        body = normalize_body(event.body or "", is_edit=is_edit, new_body=new_body)
        prompt = extract_prompt(
            body,
            command_prefix=mb.command_prefix,
            is_dm=is_direct_room(room.member_count),
        )
        if prompt is None:
            return

        session_id = conversation_id(room.room_id)
        try:
            await client.room_typing(room.room_id, typing_state=True)
            if prompt.strip() == "/reset":
                await registry.reset(session_id)
                await _send_text(client, room.room_id, "Session reset.")
                return
            if prompt.strip() == "/stats":
                text = await registry.stats(session_id, cfg)
                await _send_text(client, room.room_id, text)
                return

            from pawn_agent.core.agent_runner import run_agent_turn

            # source="matrix" tags agent_runs; session_id is a chat key, not diarization.
            result = await run_agent_turn(
                cfg=cfg,
                registry=registry,
                prompt=prompt,
                session_id=session_id,
                source="matrix",
            )
            await _send_text(client, room.room_id, result.response or "(empty reply)")
        except Exception:
            logger.exception("Matrix agent turn failed room=%s", room.room_id)
            try:
                await _send_text(client, room.room_id, "Sorry — something went wrong.")
            except Exception:
                logger.exception("Failed to send error reply")
        finally:
            try:
                await client.room_typing(room.room_id, typing_state=False)
            except Exception:
                pass

    async def on_invite(room: Any, event: Any) -> None:
        # InviteMemberEvent fires for every member in rooms.invite; only our invite.
        if event.state_key != client.user_id:
            return
        inviter = getattr(room, "inviter", None) or event.sender
        if mb.inviters and inviter not in mb.inviters:
            logger.warning("Ignoring invite to %s from %s", room.room_id, inviter)
            return
        from nio import JoinError

        for attempt in range(3):
            result = await client.join(room.room_id)
            if not isinstance(result, JoinError):
                logger.info("Joined %s", room.room_id)
                return
            logger.warning("Join %s attempt %d: %s", room.room_id, attempt + 1, result.message)
        logger.error("Unable to join %s", room.room_id)

    async def on_megolm(room: Any, event: Any) -> None:
        logger.warning("Cannot decrypt event in %s", room.room_id)
        try:
            await _send_text(
                client,
                room.room_id,
                "I couldn't decrypt that message. You may need to verify this device.",
            )
        except Exception:
            pass

    client.add_to_device_callback(
        on_to_device, (KeyVerificationEvent, UnknownToDeviceEvent)
    )
    client.add_event_callback(on_message, (RoomMessageText,))
    client.add_event_callback(on_invite, (InviteMemberEvent,))
    client.add_event_callback(on_megolm, (MegolmEvent,))


# ── Sync / presence ───────────────────────────────────────────────────────────


def next_sync_backoff(seconds: float, *, max_s: float = _SYNC_BACKOFF_MAX_S) -> float:
    """Exponential backoff cap for sync reconnect sleeps."""
    return min(max(seconds, _SYNC_BACKOFF_START_S) * 2.0, max_s)


async def _assert_online(client: Any) -> None:
    """Best-effort presence refresh so Element stays green between turns."""
    try:
        await client.set_presence("online")
    except Exception:
        logger.debug("Matrix set_presence(online) failed", exc_info=True)


async def run_sync_with_reconnect(
    client: Any,
    *,
    timeout_ms: int = _SYNC_TIMEOUT_MS,
    max_backoff_s: float = _SYNC_BACKOFF_MAX_S,
) -> None:
    """Keep ``sync_forever`` running with online presence and reconnect.

    Failed ``/sync`` bodies (nio ``SyncError`` / ``next_batch`` warnings) do not
    crash the loop, but they stop refreshing presence — so we log them and
    re-assert online. Transport crashes restart sync with backoff.
    """
    from nio import SyncError, SyncResponse

    async def on_sync_response(response: Any) -> None:
        if isinstance(response, SyncError):
            logger.warning(
                "Matrix sync error (presence may go offline): %s",
                getattr(response, "message", response),
            )
            await _assert_online(client)
        elif isinstance(response, SyncResponse):
            logger.debug("Matrix sync ok")

    client.add_response_callback(on_sync_response, (SyncResponse, SyncError))

    backoff = _SYNC_BACKOFF_START_S
    while True:
        try:
            await _assert_online(client)
            logger.info(
                "Matrix sync starting (presence=online, timeout=%sms)", timeout_ms
            )
            await client.sync_forever(
                timeout=timeout_ms,
                full_state=True,
                set_presence="online",
            )
            logger.info("Matrix sync_forever stopped")
            return
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "Matrix sync crashed; reconnecting in %.0fs", backoff
            )
            await asyncio.sleep(backoff)
            backoff = next_sync_backoff(backoff, max_s=max_backoff_s)


# ── Entrypoint ────────────────────────────────────────────────────────────────


async def start_matrix_bot(cfg: Any) -> None:
    """Login, sync forever, and route eligible messages to the sallm agent."""
    _require_nio()
    _patch_nio_sas_for_element()
    mb = cfg.matrix_bot
    _validate_cfg(mb)

    from pawn_agent.core.sallm_registry import SallmSessionRegistry

    registry = SallmSessionRegistry()
    client = _build_client(mb)
    try:
        await _login(client, mb)
        _register_callbacks(client, cfg, registry)
        logger.info("Matrix bot logged in as %s", mb.user_id)
        await run_sync_with_reconnect(client)
    finally:
        await client.close()
