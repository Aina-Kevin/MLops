import subprocess
import sys


def run_command(command):
    print()
    print("=" * 60)
    print("COMMANDE :", " ".join(command))
    print("=" * 60)

    result = subprocess.run(command)

    if result.returncode != 0:
        print()
        print("❌ ÉCHEC")
        print("La commande a échoué :", " ".join(command))
        sys.exit(result.returncode)

    print()
    print("✅ SUCCÈS")


print()
print("==============================================")
print("       PIPELINE MLOps AUTOMATIQUE")
print("==============================================")

# ==========================================
# 1. REPRODUIRE LE PIPELINE DVC
# ==========================================

run_command([
    "dvc",
    "repro"
])

# ==========================================
# 2. ÉVALUER ET PROMOUVOIR LE MODÈLE
# ==========================================

run_command([
    sys.executable,
    "src/promote.py"
])

# ==========================================
# 3. FAIRE UNE PRÉDICTION
# ==========================================

run_command([
    sys.executable,
    "src/predict.py"
])

print()
print("==============================================")
print("       PIPELINE TERMINÉ")
print("==============================================")