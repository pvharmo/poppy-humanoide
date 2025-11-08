# poppy-humanoide

# Tâches
- Intégrer le simulateur avec la boucle d'entrainement
- Créer l'environnement d'entrainement à l'aide de Gymnasium
	- Envoyer les commandes au robot
	- Recevoir les lectures des sensors
- Créer la boucle d'entrainement à l'aide de stable-baselines3 (ou SBX qui est plus rapide, mais a moins de fonctionnalités)
	- Identifier l'algorithme d'apprentissage approprié
- Identifier les rewards
- Automatic Hyperparameter Tuning (?) https://araffin.github.io/post/hyperparam-tuning/
- Logging pour suivre les rewards (avec visualisation)
- Faire un environnement de test pour l'évaluation

# Ressources
- [Reinforcement Learning Tips and Tricks](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html)
- [Stable Baselines3 RL tutorial](https://github.com/araffin/rl-tutorial-jnrr19)
- [Automatic Hyperparameter Tuning](https://araffin.github.io/post/hyperparam-tuning/)
- [SBX: réimplémentation de stable-baseline3 en Jax](https://github.com/araffin/sbx): Plus rapide (jusqu'à 20x), mais moins de fonctionnalités
- [Stable-Baselines3 - Contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib): Algorithmes d'apprentissage additionnels
## CoppeliaSim
- [Remote API (deprecated)](https://manual.coppeliarobotics.com/en/remoteApiFunctions.htm): Poppy-Humanoid utilise la version deprecated du remote API de CoppeliaSim, c'est ce qu'on utilise dans le projet
## Environnement RL
- [Documentation de Gymnasium](https://gymnasium.farama.org/)
- [Using Custom Environments](https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html#using-custom-environments "Link to this heading")
- [Exemple d'environnement custom](https://colab.research.google.com/github/araffin/rl-tutorial-jnrr19/blob/master/5_custom_gym_env.ipynb)
## Reward engineering
- [DeepMimic: Example-Guided Deep Reinforcement Learning of Physics-Based Character Skills](https://xbpeng.github.io/projects/DeepMimic/index.html)
- [Exemples de rewards pour l'environnement Humanoid de Gymnasium](https://gymnasium.farama.org/environments/mujoco/humanoid/#rewards)
## Évaluation
- [Empirical Design in Reinforcement Learning](https://arxiv.org/abs/2304.01315)
- [Deep Reinforcement Learning that Matters](https://arxiv.org/abs/1709.06560)
- [Rliable: Better Evaluation for Reinforcement Learning](https://araffin.github.io/post/rliable/)
