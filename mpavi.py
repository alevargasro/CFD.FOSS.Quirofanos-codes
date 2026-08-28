#!/usr/bin/env python3
import os
import subprocess
from pathlib import Path

def convert_avi_to_mp4_and_gif(directory="."):
    folder = Path(directory)
    avi_files = list(folder.glob("*.avi"))

    if not avi_files:
        print("❌ No se encontraron archivos .avi en el directorio.")
        return

    print(f"🔍 Se encontraron {len(avi_files)} archivo(s) .avi para procesar.\n")

    for index, avi_path in enumerate(avi_files, start=1):
        mp4_path = avi_path.with_suffix(".mp4")
        gif_path = avi_path.with_suffix(".gif")

        print(f"==================================================")
        print(f"🎬 [{index}/{len(avi_files)}] Procesando: {avi_path.name}")
        print(f"==================================================")

        # 1. Convertir AVI a MP4 (Codec H.264 para máxima compatibilidad)
        print("➡️  Convirtiendo a MP4...")
        cmd_mp4 = [
            "ffmpeg", "-y",
            "-i", str(avi_path),
            "-c:v", "libx264",
            "-crf", "20",
            "-preset", "fast",
            "-pix_fmt", "yuv420p",
            str(mp4_path)
        ]

        result_mp4 = subprocess.run(cmd_mp4, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if result_mp4.returncode != 0:
            print(f"❌ Error convirtiendo {avi_path.name} a MP4.")
            continue
        print(f"✅ MP4 generado: {mp4_path.name}")

        # 2. Convertir MP4 a GIF (Generando paleta de colores para alta calidad)
        print("➡️  Generando GIF de alta calidad...")
        cmd_gif = [
            "ffmpeg", "-y",
            "-i", str(mp4_path),
            "-vf", "fps=15,scale=800:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
            str(gif_path)
        ]

        result_gif = subprocess.run(cmd_gif, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if result_gif.returncode != 0:
            print(f"❌ Error creando GIF para {mp4_path.name}.")
            continue
        print(f"✅ GIF generado: {gif_path.name}\n")

    print("🎉 ¡Proceso finalizado exitosamente!")

if __name__ == "__main__":
    convert_avi_to_mp4_and_gif()