"""S3-compatible storage support for Pawn Diarize.

This module provides transparent download of audio files stored in
S3-compatible object storage (AWS S3, MinIO, etc.).  Any command that
accepts an audio path can be given an ``s3://`` URI instead of a local
path; the file is downloaded to a temporary location before processing
and the temp file is removed when the operation completes.

Supported URI formats
---------------------
Both of the following forms are accepted and auto-detected:

1. **Key-only** (bucket is always taken from the YAML config)::

       s3://recordings/2024/audio.wav

2. **Standard AWS URI** (bucket name embedded in the URI)::

       s3://my-bucket/recordings/2024/audio.wav

   Detection: if the first path segment of the URI matches the configured
   bucket name the URI is treated as the standard ``s3://<bucket>/<key>``
   form; otherwise the entire path after ``s3://`` is used as the object
   key on the configured bucket.

Configuration
-------------
Add an ``s3:`` section to ``.pawn-diarize.yml``:

.. code-block:: yaml

    s3:
      bucket: my-audio-bucket
      endpoint_url: https://s3.amazonaws.com
      access_key: AKIAIOSFODNN7EXAMPLE
      secret_key: wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
      region: us-east-1      # optional
      prefix: ""             # optional object key prefix for uploads
      verify_ssl: true       # optional, default true
      path_style: true       # optional, default true (use path-style addressing)
"""

import contextlib
import fnmatch
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import boto3
from botocore.config import Config


# ──────────────────────────────────────────────────────────────────────────────
# URI helpers
# ──────────────────────────────────────────────────────────────────────────────

def is_s3_path(path: str) -> bool:
    """Return *True* when *path* starts with the ``s3://`` scheme.

    Args:
        path: File path or URI string to test.

    Returns:
        ``True`` if the path is an S3 URI, ``False`` otherwise.
    """
    return path.startswith("s3://")


def parse_s3_uri(
    uri: str,
    configured_bucket: Optional[str] = None,
) -> Tuple[str, str]:
    """Parse an S3 URI and return ``(bucket, object_key)``.

    Two URI forms are supported (auto-detection):

    * ``s3://key/path.wav`` – the whole path after ``s3://`` is the object
      key; *configured_bucket* is used as the bucket name.
    * ``s3://bucket-name/key/path.wav`` – standard AWS form; if the first
      path segment matches *configured_bucket* (or when *configured_bucket*
      is ``None``) the first segment is used as the bucket name.

    Args:
        uri: S3 URI string, e.g. ``s3://my-bucket/audio/file.wav``.
        configured_bucket: Bucket name from ``.pawn-diarize.yml``, used when the
            URI does not embed a bucket name.

    Returns:
        Tuple of ``(bucket, object_key)``.

    Raises:
        ValueError: If no bucket can be resolved and *configured_bucket* is
            ``None``, or if the URI is malformed.
    """
    if not uri.startswith("s3://"):
        raise ValueError(f"Not a valid S3 URI: {uri!r}")

    # Strip scheme and split on the first '/'
    remainder = uri[len("s3://"):]
    if "/" in remainder:
        first_segment, rest = remainder.split("/", 1)
    else:
        first_segment = remainder
        rest = ""

    # If the first segment matches the configured bucket (or there is no
    # configured bucket), treat this as the standard s3://<bucket>/<key> form.
    if first_segment and (
        configured_bucket is None or first_segment == configured_bucket
    ):
        bucket = first_segment
        object_key = rest
    else:
        # The entire remainder is the object key; use the configured bucket.
        if configured_bucket is None:
            raise ValueError(
                f"Cannot resolve bucket for URI {uri!r}: no S3 bucket is "
                "configured.  Add an 's3.bucket' entry to .pawn-diarize.yml."
            )
        bucket = configured_bucket
        object_key = remainder

    if not object_key:
        raise ValueError(f"S3 URI has no object key: {uri!r}")

    return bucket, object_key


# ──────────────────────────────────────────────────────────────────────────────
# S3Config dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class S3Config:
    """Configuration for an S3-compatible storage backend.

    Mirrors the ``S3Config`` dataclass in ``pawn-diarize-recorder`` so that the
    same ``.pawn-diarize.yml`` structure works for both packages.
    """

    bucket: str
    endpoint_url: str
    access_key: str
    secret_key: str
    region: Optional[str] = None
    prefix: str = ""
    verify_ssl: bool = True
    path_style: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "S3Config":
        """Build and validate an :class:`S3Config` from a mapping.

        Args:
            data: Dictionary of S3 configuration values, typically read
                from the ``s3:`` section of ``.pawn-diarize.yml``.

        Returns:
            Validated :class:`S3Config` instance.

        Raises:
            ValueError: If any required field is missing or empty.
        """
        required_fields = ("bucket", "endpoint_url", "access_key", "secret_key")
        missing = [f for f in required_fields if not data.get(f)]
        if missing:
            raise ValueError(
                f"Missing required S3 configuration fields: {', '.join(missing)}"
            )

        return cls(
            bucket=str(data["bucket"]),
            endpoint_url=str(data["endpoint_url"]),
            access_key=str(data["access_key"]),
            secret_key=str(data["secret_key"]),
            region=str(data["region"]) if data.get("region") else None,
            prefix=str(data.get("prefix", "")),
            verify_ssl=bool(data.get("verify_ssl", True)),
            path_style=bool(data.get("path_style", True)),
        )


# ──────────────────────────────────────────────────────────────────────────────
# S3Client
# ──────────────────────────────────────────────────────────────────────────────

class S3Client:
    """Client for S3-compatible object storage.

    Wraps ``boto3`` to provide download and bucket-health-check operations
    used by Pawn Diarize when resolving ``s3://`` audio paths.

    Args:
        config: Validated :class:`S3Config` instance.
    """

    def __init__(self, config: S3Config) -> None:
        self._config = config
        addressing_style = "path" if config.path_style else "virtual"

        self._client = boto3.client(
            "s3",
            endpoint_url=config.endpoint_url,
            aws_access_key_id=config.access_key,
            aws_secret_access_key=config.secret_key,
            region_name=config.region,
            verify=config.verify_ssl,
            config=Config(s3={"addressing_style": addressing_style}),
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "S3Client":
        """Build an :class:`S3Client` directly from a configuration mapping.

        Args:
            data: Dictionary of S3 configuration values.

        Returns:
            Configured :class:`S3Client` instance.
        """
        return cls(S3Config.from_dict(data))

    @property
    def bucket(self) -> str:
        """Return the configured default bucket name."""
        return self._config.bucket

    def check_bucket(self) -> bool:
        """Verify that the configured bucket is accessible.

        Issues a lightweight ``head_bucket`` request.  Returns ``True`` when
        the bucket exists and credentials are valid; any error returns
        ``False`` without raising.

        Returns:
            ``True`` if the bucket is accessible, ``False`` otherwise.
        """
        try:
            self._client.head_bucket(Bucket=self._config.bucket)
            return True
        except Exception:
            return False

    def download_file(
        self,
        object_key: str,
        local_path: str,
        bucket: Optional[str] = None,
    ) -> None:
        """Download an S3 object to a local file.

        Args:
            object_key: Key of the object to download.
            local_path: Destination path on the local filesystem.
            bucket: Bucket to download from.  When ``None`` the configured
                default bucket is used.

        Raises:
            botocore.exceptions.ClientError: On S3 errors (not found,
                access denied, etc.).
        """
        target_bucket = bucket if bucket is not None else self._config.bucket
        self._client.download_file(target_bucket, object_key, local_path)


# ──────────────────────────────────────────────────────────────────────────────
# Context manager: resolve s3:// paths to temporary local files
# ──────────────────────────────────────────────────────────────────────────────

def expand_s3_glob(uri: str, client: "S3Client") -> List[str]:
    """Expand an S3 URI that contains shell-style wildcards.

    Wildcards (``*``, ``?``, ``[seq]``) in the object-key portion of *uri*
    are matched against the actual keys in the bucket using
    :func:`fnmatch.fnmatch`.  The prefix up to the first wildcard character
    is used as the ``Prefix`` parameter of ``list_objects_v2`` so that only
    a relevant subset of keys is fetched from S3.

    When *uri* contains no wildcard characters the function returns
    ``[uri]`` unchanged, so it is safe to call unconditionally.

    Args:
        uri: S3 URI, e.g. ``s3://bucket/audio/260224_*.flac``.
        client: Configured :class:`S3Client` used for the listing call.

    Returns:
        Sorted list of matching ``s3://<bucket>/<key>`` URIs.  Sorting is
        lexicographic so numeric suffixes (``_01``, ``_02`` …) come out in
        the expected order.

    Raises:
        ValueError: If the URI is malformed.
    """
    WILDCARDS = ("*", "?", "[")
    if not any(c in uri for c in WILDCARDS):
        return [uri]

    bucket, key_pattern = parse_s3_uri(uri, configured_bucket=client.bucket)

    # Derive the static prefix that precedes the first wildcard so we can
    # ask S3 to filter server-side before we apply fnmatch locally.
    prefix_end = min(
        (key_pattern.find(c) for c in WILDCARDS if c in key_pattern),
        default=len(key_pattern),
    )
    list_prefix = key_pattern[:prefix_end]

    boto_client = client._client  # noqa: SLF001
    paginator = boto_client.get_paginator("list_objects_v2")

    matches: List[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=list_prefix):
        for obj in page.get("Contents") or []:
            key = obj["Key"]
            if fnmatch.fnmatch(key, key_pattern):
                matches.append(f"s3://{bucket}/{key}")

    return sorted(matches)


@contextlib.contextmanager
def s3_audio_paths(
    paths: List[str],
    client: S3Client,
) -> Generator[List[str], None, None]:
    """Context manager that downloads S3 paths to temporary local files.

    Paths that are already local are passed through unchanged.  All
    temporary files are unconditionally deleted when the context exits,
    including on exception.

    Args:
        paths: List of file paths, which may include ``s3://`` URIs.
        client: Configured :class:`S3Client` used for downloads.

    Yields:
        Resolved list of local file paths in the same order as *paths*.

    Example::

        with s3_audio_paths(["s3://my-bucket/audio.wav", "local.wav"], client) as local:
            engine.transcribe(local)
    """
    temp_files: List[tempfile.NamedTemporaryFile] = []  # type: ignore[type-arg]
    resolved: List[str] = []

    try:
        for path in paths:
            if not is_s3_path(path):
                resolved.append(path)
                continue

            # Determine bucket and key
            bucket, key = parse_s3_uri(path, configured_bucket=client.bucket)

            # Preserve the original file extension so downstream audio
            # libraries (soundfile, torchaudio) can sniff the format.
            suffix = Path(key).suffix or ".audio"
            tmp = tempfile.NamedTemporaryFile(
                suffix=suffix, delete=False, prefix="pawn_diarize_s3_"
            )
            tmp.close()
            temp_files.append(tmp)

            client.download_file(key, tmp.name, bucket=bucket)
            resolved.append(tmp.name)

        yield resolved

    finally:
        for tmp in temp_files:
            try:
                Path(tmp.name).unlink(missing_ok=True)
            except Exception:
                pass
