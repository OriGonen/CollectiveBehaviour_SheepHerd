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
Possible expansions:
- Add an additional herding dog.
- Add different types of "personality" for sheep (e.g., 'courageous' sheep which stray less from the dog).
- Add physical obstacles in the field.
- Explore how much of a factor is the collective age and weight of the herd.
- Analyze selfish herd behaviour (when certain selfish/dominant sheep want to be in the middle and push inferior sheep to the outside)
We will analyze the behaviour of the expanded systems, and check whether they align with our expectations and theoretical knowledge.

Throughout the semester we will write the reports in accordance to our milestones.
After finishing and polishing our final report, we will prepare the presentation to present in class.

## Milestones:
- First report 16.11.2025
	- Create basic visualization
	- Reimplement the original code in Python
- Second report 7.12.2025
	- TBD
- Final report 11.1.2026
	- TBD

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
python simulation.py
```