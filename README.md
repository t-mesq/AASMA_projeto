# AASMA
AASMA project – 
Implementation of a single-agent system using Q-learning to solve a MDVRP, where children from different schools must return home by bus. The goal is to get the most efficient route(s) for a bus(es) to deliver each child to their homes.

How to run; 

pip install -r requirements 		#instals the necessary libraries
python open_route_service.py 		#creates a map with a fixed number of students, schools, bus ocupancy and adresses
python run_problem.py generated_map.txt #runs the code with the generated map