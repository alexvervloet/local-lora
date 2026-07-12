# Local LoRA: Fine-Tune Real Weights on a Laptop, and Prove It Helped

*This is the textbook companion for the local-lora project. The [README](README.md) tells you how to run it; [PLAN.md](PLAN.md) holds the phase definitions and the results tables; [LESSONS.md](LESSONS.md) logs what the process taught. This piece is the lecture: what LoRA actually does, why the eval gate matters more than the loss curve, and the surprising result about model size that fell out of doing this for real. It assumes you understand the idea of fine-tuning (teaching a model a behavior by showing it examples) and want to see it done on actual weights rather than a simulator.*

---

## 1. Why do it for real

Most introductions to fine-tuning, sensibly, use a simulated trainer. You learn the discipline (build a dataset, validate it, train, and gate on a held-out set) without waiting on a real training run or paying for hardware. The simulation teaches the shape of the workflow, and the shape is most of the lesson. But a simulated trainer cannot teach you the things that only appear when real weights actually move: what a loss curve looks like when it lies to you, what overfitting feels like as a number, how a small model behaves differently from a large one after the same training. Those are the lessons this project exists to surface, by swapping the mock for the real thing.

The real thing here is a LoRA fine-tune of a small open model, trained entirely on an Apple Silicon laptop using MLX (Apple's machine-learning framework for its own chips), evaluated with the same held-out-set discipline the simulation teaches. It costs nothing. The laptop's unified memory does the training, no API key is used anywhere, and no data leaves the machine. That "zero cost, fully local" property is not just convenient; it is what makes fine-tuning something you can *play* with, running the same recipe on different model sizes and different iteration counts to watch the tradeoffs move, which turns out to be where the real understanding lives.

The specific claim being tested is narrow and falsifiable, which is exactly what you want from an experiment: a LoRA-tuned model in the 1-to-3-billion-parameter range will follow a rigid support-triage format (reply as `category: <x> | reply: <one sentence>`) better than its untuned base model does. Not "feels better." Measured, on rows the model never saw during training.

## 2. What LoRA actually changes

Before the results, a word on the technique, because its cleverness explains why this is possible on a laptop at all.

A full fine-tune updates every weight in the model. For a 3-billion-parameter model, that means holding billions of numbers plus their gradients and optimizer state in memory at once, which is why full fine-tuning normally needs serious hardware. LoRA, which stands for low-rank adaptation, sidesteps this with a mathematical observation: the *change* a fine-tune makes to a model, while the model itself is enormous, can be captured in a much smaller set of numbers. So LoRA freezes the entire original model and trains only small "adapter" matrices bolted alongside it. You end up training a fraction of a percent of the parameters, which fits on modest hardware, and the resulting adapter is a small file you can keep separate from the base model, swap in and out, or later merge back in permanently.

This is what makes the whole project feasible on unified laptop memory, and it is also why the economics of fine-tuning changed for everyone when LoRA arrived: it moved fine-tuning from a data-center activity to something a single person can do on the machine in front of them. Everything below runs on that foundation.

## 3. The gate, and why the loss curve cannot be trusted

The most important number in this project is not the training loss. It is the score on a held-out set, and the gap between those two things is the central lesson.

Here is what happened on the full run. Training loss fell from 4.67 all the way to 0.028 on a training set of sixteen examples. Read on its own, that number is nearly worthless, because it is exactly what you would see whether the model *learned the task* or merely *memorized the sixteen rows*, and you cannot tell which from the training curve. A falling loss curve feels like progress and proves nothing. The only figure that settled the question was the held-out gate: the untuned base model scored 60 percent on format and 20 percent on category accuracy; the tuned adapter scored 100 percent and 80 percent, on ten rows it had never trained on. That gap is the entire claim. It is the difference between "the loss went down" and "the model got better at the job," and only the second one is a result.

This is why the project's governing rule is that a fine-tune which does not provably beat its base does not ship, and why the rule needs stating at all: because the loss curve is so seductive. It moves in the right direction, it looks like the model is learning, and it is measuring the wrong thing. Training loss is a convergence check (is the optimizer working?), not a result (did the model learn the task?). Never let one stand in for the other. The score on data the model did not see is the thing you actually care about, and making that comparison the gate is what keeps you honest.

## 4. Split the metric, or the average hides the work

The task has two parts, and scoring them separately is what made the whole thing legible.

Format adherence asks: does the reply match the required shape, `category: <x> | reply: <one sentence>`? Category accuracy asks the harder question: is the category actually correct? When you measure these separately, a clear picture emerges. Format is learned almost instantly (it hits 100 percent by 25 iterations of training and never moves again), because the shape of the output is a mechanical pattern the model picks up immediately. Category accuracy is the real work: it climbs slowly and plateaus around 80 percent, because getting the category right requires understanding the ticket, not just imitating a format.

A single blended "accuracy" number would have averaged a solved sub-task with an unsolved one into a middling figure that told you nothing about where to push. The lesson generalizes well beyond this task: **when a job bundles a mechanical part and a semantic part, measure them on separate axes.** The average of "instant" and "hard" is a number that describes neither, and it will hide exactly the information you need to decide what to do next.

## 5. More training is not more skill

The project sweeps the number of training iterations specifically to find where the win saturates, and the answer is a clean demonstration of overfitting as a number rather than a warning in a textbook.

| Iterations | Validation loss | Format | Category |
|---|---|---|---|
| base | n/a | 60% | 20% |
| 25 | 0.708 | 100% | 90% |
| 50 | 0.763 | 100% | 30% |
| 100 | 0.835 | 100% | 80% |
| 200 | 0.838 | 100% | 80% |
| 400 | 0.884 | 100% | 80% |

Two things are happening here at once. First, the held-out performance saturates almost immediately: format is maxed by iteration 25, category reaches its plateau by around 100, and nothing after that improves the model on data it did not see. Second, and this is the overfitting made concrete, the validation loss *rises* as training continues (0.708 at iteration 25, up to 0.884 at 400) even while training loss keeps falling. That divergence is the definition of overfitting: the extra iterations are fitting the sixteen training rows ever more tightly and buying nothing in generalization. The shipped 200-iteration run was already well past the useful point; around 100 iterations would have produced the same model for a fraction of the compute.

The takeaway is a practical stopping rule you can carry to any training job: **when training loss falls while validation loss rises, stop.** The curves are telling you the exact iteration where learning turned into memorizing. Extra training tokens buy a tighter fit to the training set, not more skill on the world.

## 6. On a tiny eval set, measure the noise before you rank

Look again at that sweep table, at the 50-iteration row: category accuracy reads 30 percent, a seemingly catastrophic collapse between the 90 percent at 25 iterations and the 80 percent at 100. It is not a collapse. It is noise, and understanding why is its own lesson.

The held-out set is ten rows. That means each single row is worth ten percentage points, and every score is quantized to the nearest 10 percent. A 10-point difference between two configurations is one row flipping, which is well inside the random variation you would expect from a ten-row sample. The 30 percent at 50 iterations and the 90 percent at 25 are both inside that noise; neither is a trustworthy ranking of one configuration over another. The signals you *could* trust were the ones that held across every configuration (format pinned at 100 percent) or moved monotonically (validation loss climbing steadily).

This is a discipline worth building a reflex around: **before you declare one configuration better than another, work out your measurement's resolution.** On a ten-row eval, a ten-point difference is a rounding artifact, not a finding. Rank on signals that survive the noise, and never crown a "best" configuration from a single run of a single noisy point. It is the same statistical humility that serious evaluation work insists on, made vivid by a set small enough that you can see each row's ten-point weight with your own eyes.

## 7. Fine-tuning pays the most where the base is weakest

The sharpest result in the project came from repeating the experiment on a smaller base model. Compare the 1-billion-parameter model against the 3-billion one, same training recipe:

| Model | Variant | Format | Category |
|---|---|---|---|
| 3B | base | 60% | 20% |
| 3B | + LoRA | 100% | 80% |
| 1B | base | 0% | 0% |
| 1B | + LoRA | 100% | 80% |

The untuned 1-billion model scores zero on both axes; it cannot even emit the required format, let alone pick the right category. But the *same* 200-iteration LoRA recipe lifts it to 100 percent and 80 percent, identical to the tuned 3-billion model. So the larger model gained 40 and 60 points, while the smaller model gained 100 and 80, and on this narrow task a tuned 1-billion model fully erases the advantage of a 3-billion model at a third of the parameters.

The interpretation is genuinely useful for anyone deciding which model to fine-tune. A large base model spends much of its capacity being *generally* capable across a huge range of tasks. When your task is narrow and well-specified, most of that general capacity is irrelevant, and a small model can recover the specific skill you need through fine-tuning, matching a large model on the one thing you actually asked for. **Do not reach for the bigger base by reflex.** Fine-tuning returns the most where the base is worst, and for a single well-defined task a tuned small model can match a tuned large one, which is worth measuring before you pay for parameters you may not need. Serving a 1-billion model is cheaper and faster than serving a 3-billion one, forever, so this is not an academic point.

## 8. Two habits that keep the experiment honest

Two engineering practices underpin everything above, and both are worth stealing.

The first is making the train/eval boundary structural rather than disciplined. The gate is only honest if no eval row ever leaked into training, and the way you guarantee that is not by remembering to be careful. It is by physical separation: the data-prep script only ever reads the training file, and the evaluation script reads the eval file straight from its own separate location, which the training code never touches. There is no step that *could* copy an eval row into training, so leakage is not a risk to vigilantly avoid; it is structurally impossible. The principle generalizes to any correctness property that depends on two things never mixing: **separate them physically so that mixing them is not an available move.** A boundary you have to remember to honor is a boundary you will eventually forget.

The second is treating the smoke run, not the version numbers, as the source of truth about a shifting toolchain. A self-hosted machine-learning stack moves fast and breaks silently. Over this project, the training library renamed its command-line interface between versions, and a later version crashed on import when paired with a newer version of a dependency, forcing a specific version pin. None of that is visible from reading documentation; it surfaces only when a real command actually executes. The same "observe, do not assume" discipline applied at the very end: merging the trained adapter permanently into the base model can produce weights that load without error but did not actually absorb the adaptation, so the merged model was verified by pushing a real ticket through it and checking that the output came back in the correct format. That output is the proof the merge worked. A clean exit code is not. **On a pinned, self-hosted stack, trust the smoke run and a real end-to-end generation over the documentation and the exit status.**

## 9. Where this leaves you

The headline result is clean and honest: on a narrow support-triage task, a LoRA fine-tune of a small model, trained for free on a laptop, lifted format adherence from 60 to 100 percent and category accuracy from 20 to 80, and a tuned 1-billion model matched a tuned 3-billion one. Every one of those numbers came from a held-out set, gated behind the rule that a fine-tune which does not beat its base does not ship.

But the deliverable was never really the model. It was the set of habits that made the model's improvement believable: gate on held-out data because the loss curve lies, split metrics so the average cannot hide the hard part, measure your noise before you rank configurations, stop when validation loss turns around, and make your correctness boundaries structural. Those habits are what separate a fine-tune you can trust from a loss curve that made you feel good, and they transfer to any experiment where it is easy to fool yourself about whether something actually helped, which is nearly all of them.

---

*Run it: [README.md](README.md) · Phase definitions and results: [PLAN.md](PLAN.md) · The engineering lessons in full: [LESSONS.md](LESSONS.md)*
