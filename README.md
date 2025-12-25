# CollectiveBehaviour_SheepHerd
Our GitHub repository for Collective Behaviour course 2025-2026.

The starting point of the project is the paper Collective responses of flocking sheep (Ovis aries) to a herding dog (border collie):
 - [Online](https://www.nature.com/articles/s42003-024-07245-8).
 - [Local](./res/Collective_responses_of_flocking_sheep_to_a_herding_dog.pdf).

The GitHub repository of the paper (including the model and data) can be found here https://github.com/tee-lab/collective-responses-of-flocking-sheep-to-herding-dog.

We would like to see how sheep behave collectively in a herding setting. We will expand on the paper's presented model and compare our results to theirs.

## Collaborators (Group E)
| name | github username |
|------|-----------------|
| Ori Gonen | _OriGonen_ |
| Marko Muc | _MarkoMuc_ |
| Jan Flajžík | _JanFlajz_ |


## Our Plan Throughout the Course
Review of concepts presented in the paper and their models. Expand on the model provided by the article.
After review of existing concepts, we decided to implement 2 other herding algorithms

Finally, we measure perform numerous experiments to examine the effectiveness of each algorithm based on flock properties
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
	- The second report is available [here](https://github.com/OriGonen/CollectiveBehaviour_SheepHerd/tree/second_report/second20report)
- Final report 11.1.2026
	- Implement basic version of sheep social groups
	- Perform all experiments with heterogenous social groups
	- Create a simulation video that shows how the algorithms behave based on flock properties

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