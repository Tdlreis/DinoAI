from gymnasium import Env  # Ambiente base do Gymnasium
from gymnasium.spaces import Box, Discrete  # Espaços de observação e ação
import numpy as np  # Manipulação de arrays numéricos
import time  # Para atrasos e temporização

from selenium import webdriver  # Automação do navegador
from selenium.webdriver.chrome.options import Options  # Configuração do navegador
from selenium.common.exceptions import WebDriverException  # Manipulação de exceções do Selenium
from selenium.webdriver.common.keys import Keys  # Simulação de pressionamento de teclas
from selenium.webdriver.common.action_chains import ActionChains  # Execução de ações encadeadas no Selenium

from stable_baselines3 import PPO  # Algoritmo PPO usado para RL
from stable_baselines3.common.callbacks import BaseCallback  # Callbacks personalizados
from stable_baselines3.common.monitor import Monitor  # Monitoramento de ambientes
from stable_baselines3.common.vec_env import SubprocVecEnv  # Execução paralela de ambientes

import os  # Manipulação de diretórios e arquivos
from datetime import datetime  # Registro de timestamps para salvar modelos

class WebGame(Env):
    def __init__(self, index, stage):
        super().__init__()

        # Setup do navegador
        self.driver = self.launch_browser(index)
        self.num_obstacles = 3  # Número de obstáculos a serem observados

        # Espaços de observação e ação
        self.observation_space = Box(low=0, high=255, shape=((2+(4*self.num_obstacles)),), dtype=np.float32)
        self.action_space = Discrete(3)  # 3 ações: pular, abaixar, nada

        self.done = False    
        self.lastScore = 0    
        self.index = index

        self.tempo_por_frame = 1.0 / 30.0
        self.depois = time.perf_counter()

        if stage == 1:
            self.resetScript =  '''
                                Runner.instance_.restart()
                                Runner.instance_.distanceRan = 500 / Runner.instance_.distanceMeter.config.COEFFICIENT;
                                Runner.instance_.currentSpeed = 10.5
                                '''
            self.startScore = 500
        elif stage == 2:
            self.resetScript =  '''
                                Runner.instance_.restart()
                                '''
            self.startScore = 0

    def step(self, action):
        # Mapeamento das ações
        action_map = {
            0: Keys.SPACE,       # Pular
            1: Keys.ARROW_DOWN,  # Abaixar
            2: 'no_op'           # Nenhuma ação
        }

        # Realizar a ação se não for 'no_op'
        if action_map[action] != 'no_op':
            ActionChains(self.driver).send_keys(action_map[action]).perform()

        # Captura o estado do jogo
        observation = self.get_observation()
        done, score = self.get_done()

        reward = score
        terminated = done
        truncated = False
        info = {}

        agora = time.perf_counter()
        tempo_decorrido = agora - self.depois
        if tempo_decorrido < self.tempo_por_frame:
            time.sleep(self.tempo_por_frame - tempo_decorrido)

        self.depois = time.perf_counter()

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, **kwargs):
        # Reinicia o jogo no navegador
        self.driver.switch_to.window(self.driver.current_window_handle)
        self.driver.execute_script('Runner.instance_.restart()')

        observation = self.get_observation()
        self.lastScore = self.startScore
        return observation, {}

    def get_observation(self):
        # Captura os dados do jogo
        data = self.get_game_data()
        obstacles = data['obstacles']

        # Lista para armazenar informações dos obstáculos
        obstacles_data = []
        for i in range(self.num_obstacles):
            if i < len(obstacles):
                obstacle = obstacles[i]
                obstacle_x = obstacle['xPos']
                obstacle_y = obstacle['yPos']
                obstacle_width = obstacle['width']
                obstacle_height = obstacle['height']
            else:
                obstacle_x, obstacle_y, obstacle_width, obstacle_height = [0]*4

            obstacles_data.extend([obstacle_x, obstacle_y, obstacle_width, obstacle_height])

        state = np.array([data["speed"], data["dinoY"]] + obstacles_data, dtype=np.float32)
        return state

    def get_done(self):
        # Verifica se o jogo terminou e captura a pontuação atual
        done = self.driver.execute_script("return Runner.instance_.crashed")
        currentScore = int(self.driver.execute_script("return Runner.instance_.distanceMeter.digits.join('')"))

        score = currentScore - self.lastScore
        if not done:
            self.lastScore = currentScore
        else:
            self.lastScore = self.startScore
            score = -100  # Penalidade ao perder
        return done, score

    def launch_browser(self, index):
        options = Options()
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-extensions")
        options.add_argument("--window-size=500,350")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")

        driver = webdriver.Chrome(options=options)

        x, y = self.get_window_position(index)
        driver.set_window_position(x, y)

        driver.set_network_conditions(
            offline=True,
            latency=5,
            download_throughput=500 * 1024,
            upload_throughput=500 * 1024,
        )

        try:
            driver.get('http://www.google.com')
        except WebDriverException:
            pass

        time.sleep(2)
        ActionChains(driver).send_keys(Keys.SPACE).perform()

        driver.execute_script("""
        document.addEventListener('keydown', function(event) {
            if (event.code === 'Space' || event.code === 'ArrowDown') {
                event.preventDefault();
            }
        });
        """)

        return driver

    def get_window_position(self, index, window_width=500, window_height=350):
        index = 5 - index
        if index < 2:
            x = index * window_width
            y = 0
        elif index < 4:
            x = (index - 2) * window_width
            y = window_height - 100
        else:
            x = (index - 4) * window_width
            y = 2 * (window_height - 100)
        return x, y

    def get_game_data(self):
        return self.driver.execute_script("""
            const runner = Runner.instance_;
            return {
                speed: runner.currentSpeed || 0,
                dinoY: runner.tRex.yPos || 0,
                obstacles: runner.horizon.obstacles.slice(0, 3).map(ob => ({
                    xPos: ob.xPos || 0,
                    yPos: ob.yPos || 0,
                    width: ob.typeConfig.width || 0,
                    height: ob.typeConfig.height || 0
                }))
            };
        """)

    def close(self):
        self.driver.quit()

class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, log_interval=1000, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.log_interval = log_interval
        self.episode_rewards = []

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            model_path = os.path.join(self.save_path, f'best_model_{self.n_calls}_{timestamp}')
            self.model.save(model_path)

        infos = self.locals.get('infos', [])
        if infos and 'episode' in infos[0]:
            episode_reward = infos[0]['episode']['r']
            self.episode_rewards.append(episode_reward)

            if len(self.episode_rewards) >= 100:
                mean_reward = np.mean(self.episode_rewards[-100:])
                self.logger.record('rollout/mean_reward', mean_reward)

                if self.verbose > 0:
                    print(f"Step {self.n_calls}: Mean Reward (últimos 100 episódios) = {mean_reward}")

        return True

if __name__ == "__main__":
    CHECKPOINT_DIR = './train/'
    LOG_DIR = './logs/'

    def make_env(index, stage):
        def _init():
            env = Monitor(WebGame(index, stage), LOG_DIR)
            return env
        return _init

    num_cpu = 6
    callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR, log_interval=1000)

    envs = SubprocVecEnv([make_env(i, 1) for i in range(num_cpu)])
    model = PPO('MlpPolicy', envs, tensorboard_log=LOG_DIR, policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])), verbose=1, device='cpu')
    model.learn(total_timesteps=2500000, callback=callback)
    model.save(os.path.join(CHECKPOINT_DIR, "ppo_dino_intermediate"))
    envs.close()
    print("Treinamento concluído estágio 1 e modelo intermediario salvo.")

    envs = SubprocVecEnv([make_env(i, 2) for i in range(num_cpu)])
    model = PPO.load(os.path.join(CHECKPOINT_DIR, "ppo_dino_intermediate"), envs, tensorboard_log=LOG_DIR, verbose=1, device='cpu')
    model.learn(total_timesteps=2500000, callback=callback)
    model.save(os.path.join(CHECKPOINT_DIR, "ppo_dino_final"))
    envs.close()
    print("Treinamento concluído estágio 2 e modelo final salvo.")

