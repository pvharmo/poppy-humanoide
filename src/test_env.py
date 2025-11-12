from environment import RobotEnv
import numpy as np


# Juste un test rapide
def quick_test():
    scene_path = "chemin/vers/votre/scene_poppy.ttt"

    try:
        env = RobotEnv(scene_path, headless=False)
        print("✅ Environnement créé avec succès")

        obs = env.reset()
        print(f"✅ Reset réussi - Observation shape: {obs.shape}")

        # Test de 10 steps
        for i in range(10):
            action = np.random.uniform(-0.5, 0.5, size=env.action_space.shape)
            obs, reward, done, info = env.step(action)
            print(f"Step {i}: reward={reward:.2f}, done={done}")

            if done:
                break

        env.close()
        print("✅ Test terminé sans erreurs")

    except Exception as e:
        print(f"❌ Erreur: {e}")


quick_test()