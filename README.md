## Instructions
1. Build the carla simulator from source so that the python api version matches your system python.
2. [Define environment variables](https://leaderboard.carla.org/get_started_v2_1/#13-define-the-environment-variables) with the following changes:
   1. `export CARLA_ROOT=<path to your compiled carla simulator>`
      1. i.e. `/home/zhuochen/Softwares/carla_0.9.15/Dist/CARLA_Shipping_0.9.15.2-dirty/LinuxNoEditor`
   2. `export LEADERBOARD_ROOT=<path to this repo>`
   3. in `PYTHONPATH`, change `"${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.14-py3.7-linux-x86_64.egg"` to the path to the corresponding python api file that you just built.
      1. i.e. `"${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.15-py3.10-linux-x86_64.egg"`
3. Follow other setup instructions in the documentation, and verify that you can run the human agent.
4. Checkout the latest [MTR-carla](https://github.com/kevin1zc/MTR-carla)
5. Use `leaderboard/autoagents/mtr_agent.py` as a starting point. You can either change it to incorporate the MPC controller, or create a new agent file.
5. If create a new file, go to `run_leaderboard.sh` and change `TEAM_AGENT` accordingly.

The main goal of the CARLA Autonomous Driving Leaderboard is to evaluate the driving proficiency of autonomous agents in realistic traffic situations. The leaderboard serves as an open platform for the community to perform fair and reproducible evaluations, simplifying the comparison between different approaches.

Autonomous agents have to drive through a set of predefined routes. For each route, agents are initialized at a starting point and have to drive to a destination point. The agents will be provided with a description of the route. Routes will happen in a variety of areas, including freeways, urban scenes, and residential districts.

Agents will face multiple traffic situations based in the NHTSA typology, such as:

* Lane merging
* Lane changing
* Negotiations at traffic intersections
* Negotiations at roundabouts
* Handling traffic lights and traffic signs
* Coping with pedestrians, cyclists and other elements

The user can change the weather of the simulation, allowing the evaluation of the agent in a variety of weather conditions, including daylight scenes, sunset, rain, fog, and night, among others.

More information can be found [here](https://leaderboard.carla.org/)