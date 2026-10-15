"""Capture une zone de l'écran, appuie sur flèche droite, recommence jusqu'à la fin.

À lancer avec le Python de Windows (pas celui de WSL) :
    pip install mss pyautogui pillow
    py capture_slices.py --out captures

1. Un voile gris couvre l'écran : tracer à la souris le rectangle à capturer.
2. Pendant le compte à rebours, cliquer sur la fenêtre de l'application pour lui donner le focus.
3. Le script capture, appuie sur →, et s'arrête quand l'image ne change plus
   (= dernière image atteinte). Ctrl+C ou souris dans un coin de l'écran pour interrompre.
"""
import argparse
import ctypes
import time
import tkinter as tk
from pathlib import Path

import mss
import mss.tools
import pyautogui


def select_region():
    """Overlay plein écran : l'utilisateur trace un rectangle, renvoie (left, top, width, height)."""
    root = tk.Tk()
    root.attributes("-fullscreen", True)
    root.attributes("-alpha", 0.3)
    root.attributes("-topmost", True)
    canvas = tk.Canvas(root, cursor="cross", bg="grey")
    canvas.pack(fill="both", expand=True)
    pts = {}

    def press(e):
        pts["x0"], pts["y0"] = e.x_root, e.y_root
        pts["rect"] = canvas.create_rectangle(e.x, e.y, e.x, e.y, outline="lime", width=2)

    def drag(e):
        canvas.coords(pts["rect"], pts["x0"] - root.winfo_rootx(), pts["y0"] - root.winfo_rooty(), e.x, e.y)

    def release(e):
        pts["x1"], pts["y1"] = e.x_root, e.y_root
        root.destroy()

    canvas.bind("<ButtonPress-1>", press)
    canvas.bind("<B1-Motion>", drag)
    canvas.bind("<ButtonRelease-1>", release)
    root.bind("<Escape>", lambda e: root.destroy())
    root.mainloop()

    if "x1" not in pts:
        raise SystemExit("Sélection annulée.")
    left, top = min(pts["x0"], pts["x1"]), min(pts["y0"], pts["y1"])
    width, height = abs(pts["x1"] - pts["x0"]), abs(pts["y1"] - pts["y0"])
    if width < 5 or height < 5:
        raise SystemExit("Zone trop petite.")
    return left, top, width, height


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="captures", help="dossier de sortie")
    p.add_argument("--delay", type=float, default=0.3, help="attente (s) après chaque appui sur →")
    p.add_argument("--countdown", type=int, default=5, help="secondes pour donner le focus à l'application")
    p.add_argument("--patience", type=int, default=3,
                   help="nombre de captures identiques d'affilée avant de considérer la fin atteinte")
    args = p.parse_args()

    # Coordonnées en pixels physiques, même avec une mise à l'échelle Windows à 125 %/150 %.
    ctypes.windll.shcore.SetProcessDpiAwareness(2)

    left, top, width, height = select_region()
    region = {"left": left, "top": top, "width": width, "height": height}
    print(f"Zone : {region}")

    for s in range(args.countdown, 0, -1):
        print(f"Cliquez sur la fenêtre de l'application... {s}")
        time.sleep(1)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    previous, identical, index = None, 0, 0

    with mss.mss() as sct:
        while identical < args.patience:
            shot = sct.grab(region)
            if shot.rgb == previous:
                identical += 1
            else:
                identical = 0
                previous = shot.rgb
                path = out / f"slice_{index:04d}.png"
                mss.tools.to_png(shot.rgb, shot.size, output=str(path))
                print(f"Enregistré : {path}")
                index += 1
            pyautogui.press("right")
            time.sleep(args.delay)

    print(f"Fin atteinte : {index} images dans {out.resolve()}")


if __name__ == "__main__":
    main()
