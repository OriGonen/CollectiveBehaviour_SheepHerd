# CollectiveBehaviour_SheepHerd
Our GitHub repository for Collective Behaviour course 2025-2026.

The starting point of the project is the paper **Collective responses of flocking sheep (Ovis aries) to a herding dog (border collie)**:
 - [Online](https://www.nature.com/articles/s42003-024-07245-8).
 - [Local](./res/Collective_responses_of_flocking_sheep_to_a_herding_dog.pdf).

The GitHub repository of the paper (including the model and data) can be found here https://github.com/tee-lab/collective-responses-of-flocking-sheep-to-herding-dog.

Our goal in the project is to study how sheep behave collectively in a herding setting by expanding on the paper's presented model.

## Collaborators (Group E)
| name | github username |
|------|-----------------|
| Ori Gonen | _OriGonen_ |
| Marko Muc | _MarkoMuc_ |
| Jan Flajžík | _JanFlajz_ |


## Our Plan Throughout the Course
Review of concepts presented in the paper and their models. Expand on the model provided by the article.
We decided to expand on the model by integrating a fatigue model: each agent has an internal fatigue state which affects their locomotor speed, while the interaction rules remain unchanged.

Finally, we perform experiments with different parameters to observe how flock dynamics change.
Throughout the semester we will write the reports in accordance to our milestones.
After finishing and polishing our final report, we will prepare the presentation to present in class.

## Milestones:
- First report 16.11.2025
	- ~~Create basic visualization~~
	- ~~Reimplement the original code in Python~~
	- The first report is available [here](https://github.com/OriGonen/CollectiveBehaviour_SheepHerd/tree/firstReport/report1)
- Second report 7.12.2025
	- ~~Implement existing sheep herding algorithms as proposed~~
	- ~~Perform basic experiments with the algorithms~~
	- The second report is available [here](https://github.com/OriGonen/CollectiveBehaviour_SheepHerd/blob/second_report/second%20report/Collective_Behavior_Group_E_Report_2.pdf)
- Final report 11.1.2026
	- ~~Implement fatigue model~~
	- ~~Conduct experiments with different parameters~~
	- ~~Create a simulation video to demonstrate~~
 	- ~~Create the presentation~~
    - The final report is available [here](https://github.com/OriGonen/CollectiveBehaviour_SheepHerd/blob/main/report/GroupE_FinalReport.pdf)

## Running the simulation

First create a virtual environment and activate it:

```bash
python -m venv .venv
source .venv/bin/activate
```
OR

```bash
uv venv
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the simulation:

```bash
python visualize.py
```
