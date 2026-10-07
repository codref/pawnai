/** Injected overlay: drag a rectangle, crop the visible-tab PNG. */

(function () {
  if (window.__pawnRegionReady) return;
  window.__pawnRegionReady = true;

  chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
    if (msg?.type !== "pawn-region-pick") return false;
    pickRegion(msg.dataUrl)
      .then((result) => sendResponse(result))
      .catch((err) => sendResponse({ ok: false, error: String(err?.message || err) }));
    return true;
  });

  function pickRegion(dataUrl) {
    return new Promise((resolve) => {
      const root = document.createElement("div");
      root.id = "pawn-region-root";
      Object.assign(root.style, {
        position: "fixed",
        inset: "0",
        zIndex: "2147483646",
        cursor: "crosshair",
        background: "rgba(0,0,0,0.15)",
      });
      const box = document.createElement("div");
      Object.assign(box.style, {
        position: "fixed",
        border: "2px solid #4a9e8e",
        background: "rgba(74,158,142,0.15)",
        display: "none",
        pointerEvents: "none",
      });
      root.appendChild(box);
      document.documentElement.appendChild(root);

      let startX = 0;
      let startY = 0;
      let dragging = false;

      const cleanup = () => {
        root.remove();
        window.removeEventListener("keydown", onKey, true);
      };

      const onKey = (ev) => {
        if (ev.key === "Escape") {
          cleanup();
          resolve({ ok: false, error: "Cancelled" });
        }
      };
      window.addEventListener("keydown", onKey, true);

      root.addEventListener("mousedown", (ev) => {
        dragging = true;
        startX = ev.clientX;
        startY = ev.clientY;
        box.style.display = "block";
        box.style.left = `${startX}px`;
        box.style.top = `${startY}px`;
        box.style.width = "0px";
        box.style.height = "0px";
      });

      root.addEventListener("mousemove", (ev) => {
        if (!dragging) return;
        const x = Math.min(startX, ev.clientX);
        const y = Math.min(startY, ev.clientY);
        const w = Math.abs(ev.clientX - startX);
        const h = Math.abs(ev.clientY - startY);
        box.style.left = `${x}px`;
        box.style.top = `${y}px`;
        box.style.width = `${w}px`;
        box.style.height = `${h}px`;
      });

      root.addEventListener("mouseup", async (ev) => {
        if (!dragging) return;
        dragging = false;
        const x = Math.min(startX, ev.clientX);
        const y = Math.min(startY, ev.clientY);
        const w = Math.abs(ev.clientX - startX);
        const h = Math.abs(ev.clientY - startY);
        cleanup();
        if (w < 4 || h < 4) {
          resolve({ ok: false, error: "Region too small" });
          return;
        }
        try {
          const data_base64 = await crop(dataUrl, x, y, w, h);
          resolve({ ok: true, data_base64 });
        } catch (err) {
          resolve({ ok: false, error: String(err?.message || err) });
        }
      });
    });
  }

  function crop(dataUrl, x, y, w, h) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        const scaleX = img.naturalWidth / window.innerWidth;
        const scaleY = img.naturalHeight / window.innerHeight;
        const sx = Math.round(x * scaleX);
        const sy = Math.round(y * scaleY);
        const sw = Math.max(1, Math.round(w * scaleX));
        const sh = Math.max(1, Math.round(h * scaleY));
        const canvas = document.createElement("canvas");
        canvas.width = sw;
        canvas.height = sh;
        const ctx = canvas.getContext("2d");
        if (!ctx) {
          reject(new Error("Canvas unavailable"));
          return;
        }
        ctx.drawImage(img, sx, sy, sw, sh, 0, 0, sw, sh);
        const out = canvas.toDataURL("image/png");
        const b64 = out.split(",", 2)[1] || "";
        resolve(b64);
      };
      img.onerror = () => reject(new Error("Failed to load screenshot"));
      img.src = dataUrl;
    });
  }
})();
