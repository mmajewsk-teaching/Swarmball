import sys
import os

# Dodajemy główny folder projektu do ścieżki Pythona, żeby importy działały
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environment.swarmball_env import SwarmBall


def test_smoke_step_and_reset():
    # [ZMIANA] Podstawowy test sprawdzający, czy reset i step nie rzucają błędów
    print("Inicjalizacja środowiska...")
    env = SwarmBall()

    print("Testowanie reset()...")
    obs, info = env.reset()
    assert isinstance(obs, dict), "Reset musi zwracać słownik (dict)"
    assert 'picture' in obs and 'thresholds' in obs, "Brakuje kluczy w obserwacji"

    print("Testowanie step()...")
    action = env.action_space.sample()  # Losowa akcja
    obs, reward, terminated, truncated, info = env.step(action)

    assert isinstance(reward, float), "Nagroda musi być typu float"
    assert isinstance(terminated, bool), "Terminated musi być typu bool"
    assert isinstance(truncated, bool), "Truncated musi być typu bool"

    print("✅ Smoke test zaliczony! Środowisko działa poprawnie z Gymnasium API.")


if __name__ == "__main__":
    test_smoke_step_and_reset()