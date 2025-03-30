# Flattened Markdown Documents from /Users/timgregg/mcp/Github/stanfordnlp/dspy/docs/docs



## File: faqs.md

---
sidebar_position: 998
---

# FAQs

## Is DSPy right for me? DSPy vs. other frameworks

The **DSPy** philosophy and abstraction differ significantly from other libraries and frameworks, so it's usually straightforward to decide when **DSPy** is (or isn't) the right framework for your usecase. If you're a NLP/AI researcher (or a practitioner exploring new pipelines or new tasks), the answer is generally an invariable **yes**. If you're a practitioner doing other things, please read on.

**DSPy vs. thin wrappers for prompts (OpenAI API, MiniChain, basic templating)** In other words: _Why can't I just write my prompts directly as string templates?_ Well, for extremely simple settings, this _might_ work just fine. (If you're familiar with neural networks, this is like expressing a tiny two-layer NN as a Python for-loop. It kinda works.) However, when you need higher quality (or manageable cost), then you need to iteratively explore multi-stage decomposition, improved prompting, data bootstrapping, careful finetuning, retrieval augmentation, and/or using smaller (or cheaper, or local) models. The true expressive power of building with foundation models lies in the interactions between these pieces. But every time you change one piece, you likely break (or weaken) multiple other components. **DSPy** cleanly abstracts away (_and_ powerfully optimizes) the parts of these interactions that are external to your actual system design. It lets you focus on designing the module-level interactions: the _same program_ expressed in 10 or 20 lines of **DSPy** can easily be compiled into multi-stage instructions for `GPT-4`, detailed prompts for `Llama2-13b`, or finetunes for `T5-base`. Oh, and you wouldn't need to maintain long, brittle, model-specific strings at the core of your project anymore.

**DSPy vs. application development libraries like LangChain, LlamaIndex** LangChain and LlamaIndex target high-level application development; they offer _batteries-included_, pre-built application modules that plug in with your data or configuration. If you'd be happy to use a generic, off-the-shelf prompt for question answering over PDFs or standard text-to-SQL, you will find a rich ecosystem in these libraries. **DSPy** doesn't internally contain hand-crafted prompts that target specific applications. Instead, **DSPy** introduces a small set of much more powerful and general-purpose modules _that can learn to prompt (or finetune) your LM within your pipeline on your data_. when you change your data, make tweaks to your program's control flow, or change your target LM, the **DSPy compiler** can map your program into a new set of prompts (or finetunes) that are optimized specifically for this pipeline. Because of this, you may find that **DSPy** obtains the highest quality for your task, with the least effort, provided you're willing to implement (or extend) your own short program. In short, **DSPy** is for when you need a lightweight but automatically-optimizing programming model — not a library of predefined prompts and integrations. If you're familiar with neural networks: This is like the difference between PyTorch (i.e., representing **DSPy**) and HuggingFace Transformers (i.e., representing the higher-level libraries).

**DSPy vs. generation control libraries like Guidance, LMQL, RELM, Outlines** These are all exciting new libraries for controlling the individual completions of LMs, e.g., if you want to enforce JSON output schema or constrain sampling to a particular regular expression. This is very useful in many settings, but it's generally focused on low-level, structured control of a single LM call. It doesn't help ensure the JSON (or structured output) you get is going to be correct or useful for your task. In contrast, **DSPy** automatically optimizes the prompts in your programs to align them with various task needs, which may also include producing valid structured outputs. That said, we are considering allowing **Signatures** in **DSPy** to express regex-like constraints that are implemented by these libraries.

## Basic Usage

**How should I use DSPy for my task?** We wrote a [eight-step guide](/building-blocks/solving_your_task) on this. In short, using DSPy is an iterative process. You first define your task and the metrics you want to maximize, and prepare a few example inputs — typically without labels (or only with labels for the final outputs, if your metric requires them). Then, you build your pipeline by selecting built-in layers (`modules`) to use, giving each layer a `signature` (input/output spec), and then calling your modules freely in your Python code. Lastly, you use a DSPy `optimizer` to compile your code into high-quality instructions, automatic few-shot examples, or updated LM weights for your LM.

**How do I convert my complex prompt into a DSPy pipeline?** See the same answer above.

**What do DSPy optimizers tune?** Or, _what does compiling actually do?_ Each optimizer is different, but they all seek to maximize a metric on your program by updating prompts or LM weights. Current DSPy `optimizers` can inspect your data, simulate traces through your program to generate good/bad examples of each step, propose or refine instructions for each step based on past results, finetune the weights of your LM on self-generated examples, or combine several of these to improve quality or cut cost. We'd love to merge new optimizers that explore a richer space: most manual steps you currently go through for prompt engineering, "synthetic data" generation, or self-improvement can probably generalized into a DSPy optimizer that acts on arbitrary LM programs.

Other FAQs. We welcome PRs to add formal answers to each of these here. You will find the answer in existing issues, tutorials, or the papers for all or most of these.

- **How do I get multiple outputs?**

You can specify multiple output fields. For the short-form signature, you can list multiple outputs as comma separated values, following the "->" indicator (e.g. "inputs -> output1, output2"). For the long-form signature, you can include multiple `dspy.OutputField`s.


- **How do I define my own metrics? Can metrics return a float?**

You can define metrics as simply Python functions that process model generations and evaluate them based on user-defined requirements. Metrics can compare existent data (e.g. gold labels) to model predictions or they can be used to assess various components of an output using validation feedback from LMs (e.g. LLMs-as-Judges). Metrics can return `bool`, `int`, and `float` types scores. Check out the official [Metrics docs](/building-blocks/5-metrics) to learn more about defining custom metrics and advanced evaluations using AI feedback and/or DSPy programs.

- **How expensive or slow is compiling??**

To reflect compiling metrics, we highlight an experiment for reference, compiling the [`SimplifiedBaleen`](/tutorials/simplified-baleen) using the [`dspy.BootstrapFewShotWithRandomSearch`](/deep-dive/teleprompter/bootstrap-fewshot) optimizer on the `gpt-3.5-turbo-1106` model over 7 candidate programs and 10 threads. We report that compiling this program takes around 6 minutes with 3200 API calls, 2.7 million input tokens and 156,000 output tokens, reporting a total cost of $3 USD (at the current pricing of the OpenAI model).

Compiling DSPy `optimizers` naturally will incur additional LM calls, but we substantiate this overhead with minimalistic executions with the goal of maximizing performance. This invites avenues to enhance performance of smaller models by compiling DSPy programs with larger models to learn enhanced behavior during compile-time and propagate such behavior to the tested smaller model during inference-time.  


## Deployment or Reproducibility Concerns

- **How do I save a checkpoint of my compiled program?**

Here is an example of saving/loading a compiled module:

```python
cot_compiled = teleprompter.compile(CoT(), trainset=trainset, valset=devset)

#Saving
cot_compiled.save('compiled_cot_gsm8k.json')

#Loading:
cot = CoT()
cot.load('compiled_cot_gsm8k.json')
```

- **How do I export for deployment?**

Exporting DSPy programs is simply saving them as highlighted above!

- **How do I search my own data?**

Open source libraries such as [RAGautouille](https://github.com/bclavie/ragatouille) enable you to search for your own data through advanced retrieval models like ColBERT with tools to embed and index documents. Feel free to integrate such libraries to create searchable datasets while developing your DSPy programs!

- **How do I turn off the cache? How do I export the cache?**

From v2.5, you can turn off the cache by setting `cache` parameter in `dspy.LM` to `False`:

```python
dspy.LM('openai/gpt-4o-mini',  cache=False)
```

Your local cache will be saved to the global env directory `os.environ["DSP_CACHEDIR"]` or for notebooks `os.environ["DSP_NOTEBOOK_CACHEDIR"]`. You can usually set the cachedir to `os.path.join(repo_path, 'cache')` and export this cache from here:
```python
os.environ["DSP_NOTEBOOK_CACHEDIR"] = os.path.join(os.getcwd(), 'cache')
```

!!! warning "Important"
    `DSP_CACHEDIR` is responsible for old clients (including dspy.OpenAI, dspy.ColBERTv2, etc.) and `DSPY_CACHEDIR` is responsible for the new dspy.LM client.

    In the AWS lambda deployment, you should disable both DSP_\* and DSPY_\*.


## Advanced Usage

- **How do I parallelize?**
You can parallelize DSPy programs during both compilation and evaluation by specifying multiple thread settings in the respective DSPy `optimizers` or within the `dspy.Evaluate` utility function.

- **How do freeze a module?**

Modules can be frozen by setting their `._compiled` attribute to be True, indicating the module has gone through optimizer compilation and should not have its parameters adjusted. This is handled internally in optimizers such as `dspy.BootstrapFewShot` where the student program is ensured to be frozen before the teacher propagates the gathered few-shot demonstrations in the bootstrapping process. 

- **How do I use DSPy assertions?**

    a) **How to Add Assertions to Your Program**:
    - **Define Constraints**: Use `dspy.Assert` and/or `dspy.Suggest` to define constraints within your DSPy program. These are based on boolean validation checks for the outcomes you want to enforce, which can simply be Python functions to validate the model outputs.
    - **Integrating Assertions**: Keep your Assertion statements following a model generations (hint: following a module layer)

    b) **How to Activate the Assertions**:
    1. **Using `assert_transform_module`**:
        - Wrap your DSPy module with assertions using the `assert_transform_module` function, along with a `backtrack_handler`. This function transforms your program to include internal assertions backtracking and retry logic, which can be customized as well:
        `program_with_assertions = assert_transform_module(ProgramWithAssertions(), backtrack_handler)`
    2. **Activate Assertions**:
        - Directly call `activate_assertions` on your DSPy program with assertions: `program_with_assertions = ProgramWithAssertions().activate_assertions()`

    **Note**: To use Assertions properly, you must **activate** a DSPy program that includes `dspy.Assert` or `dspy.Suggest` statements from either of the methods above. 

## Errors

- **How do I deal with "context too long" errors?**

If you're dealing with "context too long" errors in DSPy, you're likely using DSPy optimizers to include demonstrations within your prompt, and this is exceeding your current context window. Try reducing these parameters (e.g. `max_bootstrapped_demos` and `max_labeled_demos`). Additionally, you can also reduce the number of retrieved passages/docs/embeddings to ensure your prompt is fitting within your model context length.

A more general fix is simply increasing the number of `max_tokens` specified to the LM request (e.g. `lm = dspy.OpenAI(model = ..., max_tokens = ...`).


## End of faqs.md

---


## File: cheatsheet.md

---
sidebar_position: 999
---

# DSPy Cheatsheet

This page will contain snippets for frequent usage patterns.

## DSPy DataLoaders

Import and initializing a DataLoader Object:

```python
import dspy
from dspy.datasets import DataLoader

dl = DataLoader()
```

### Loading from HuggingFace Datasets

```python
code_alpaca = dl.from_huggingface("HuggingFaceH4/CodeAlpaca_20K")
```

You can access the dataset of the splits by calling key of the corresponding split:

```python
train_dataset = code_alpaca['train']
test_dataset = code_alpaca['test']
```

### Loading specific splits from HuggingFace

You can also manually specify splits you want to include as a parameters and it'll return a dictionary where keys are splits that you specified:

```python
code_alpaca = dl.from_huggingface(
    "HuggingFaceH4/CodeAlpaca_20K",
    split = ["train", "test"],
)

print(f"Splits in dataset: {code_alpaca.keys()}")
```

If you specify a single split then dataloader will return a List of `dspy.Example` instead of dictionary:

```python
code_alpaca = dl.from_huggingface(
    "HuggingFaceH4/CodeAlpaca_20K",
    split = "train",
)

print(f"Number of examples in split: {len(code_alpaca)}")
```

You can slice the split just like you do with HuggingFace Dataset too:

```python
code_alpaca_80 = dl.from_huggingface(
    "HuggingFaceH4/CodeAlpaca_20K",
    split = "train[:80%]",
)

print(f"Number of examples in split: {len(code_alpaca_80)}")

code_alpaca_20_80 = dl.from_huggingface(
    "HuggingFaceH4/CodeAlpaca_20K",
    split = "train[20%:80%]",
)

print(f"Number of examples in split: {len(code_alpaca_20_80)}")
```

### Loading specific subset from HuggingFace

If a dataset has a subset you can pass it as an arg like you do with `load_dataset` in HuggingFace:

```python
gms8k = dl.from_huggingface(
    "gsm8k",
    "main",
    input_keys = ("question",),
)

print(f"Keys present in the returned dict: {list(gms8k.keys())}")

print(f"Number of examples in train set: {len(gms8k['train'])}")
print(f"Number of examples in test set: {len(gms8k['test'])}")
```

### Loading from CSV

```python
dolly_100_dataset = dl.from_csv("dolly_subset_100_rows.csv",)
```

You can choose only selected columns from the csv by specifying them in the arguments:

```python
dolly_100_dataset = dl.from_csv(
    "dolly_subset_100_rows.csv",
    fields=("instruction", "context", "response"),
    input_keys=("instruction", "context")
)
```

### Splitting a List of `dspy.Example`

```python
splits = dl.train_test_split(dataset, train_size=0.8) # `dataset` is a List of dspy.Example
train_dataset = splits['train']
test_dataset = splits['test']
```

### Sampling from List of `dspy.Example`

```python
sampled_example = dl.sample(dataset, n=5) # `dataset` is a List of dspy.Example
```

## DSPy Programs

### dspy.Signature

```python
class BasicQA(dspy.Signature):
    """Answer questions with short factoid answers."""

    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")
```

### dspy.ChainOfThought

```python
generate_answer = dspy.ChainOfThought(BasicQA)

# Call the predictor on a particular input alongside a hint.
question='What is the color of the sky?'
pred = generate_answer(question=question)
```

### dspy.ChainOfThoughtwithHint

```python
generate_answer = dspy.ChainOfThoughtWithHint(BasicQA)

# Call the predictor on a particular input alongside a hint.
question='What is the color of the sky?'
hint = "It's what you often see during a sunny day."
pred = generate_answer(question=question, hint=hint)
```

### dspy.ProgramOfThought

```python
pot = dspy.ProgramOfThought(BasicQA)

question = 'Sarah has 5 apples. She buys 7 more apples from the store. How many apples does Sarah have now?'
result = pot(question=question)

print(f"Question: {question}")
print(f"Final Predicted Answer (after ProgramOfThought process): {result.answer}")
```

### dspy.ReACT

```python
react_module = dspy.ReAct(BasicQA)

question = 'Sarah has 5 apples. She buys 7 more apples from the store. How many apples does Sarah have now?'
result = react_module(question=question)

print(f"Question: {question}")
print(f"Final Predicted Answer (after ReAct process): {result.answer}")
```

### dspy.Retrieve

```python
colbertv2_wiki17_abstracts = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')
dspy.settings.configure(rm=colbertv2_wiki17_abstracts)

#Define Retrieve Module
retriever = dspy.Retrieve(k=3)

query='When was the first FIFA World Cup held?'

# Call the retriever on a particular query.
topK_passages = retriever(query).passages

for idx, passage in enumerate(topK_passages):
    print(f'{idx+1}]', passage, '\n')
```

## DSPy Metrics

### Function as Metric

To create a custom metric you can create a function that returns either a number or a boolean value:

```python
def parse_integer_answer(answer, only_first_line=True):
    try:
        if only_first_line:
            answer = answer.strip().split('\n')[0]

        # find the last token that has a number in it
        answer = [token for token in answer.split() if any(c.isdigit() for c in token)][-1]
        answer = answer.split('.')[0]
        answer = ''.join([c for c in answer if c.isdigit()])
        answer = int(answer)

    except (ValueError, IndexError):
        # print(answer)
        answer = 0

    return answer

# Metric Function
def gsm8k_metric(gold, pred, trace=None) -> int:
    return int(parse_integer_answer(str(gold.answer))) == int(parse_integer_answer(str(pred.answer)))
```

### LLM as Judge

```python
class FactJudge(dspy.Signature):
    """Judge if the answer is factually correct based on the context."""

    context = dspy.InputField(desc="Context for the prediction")
    question = dspy.InputField(desc="Question to be answered")
    answer = dspy.InputField(desc="Answer for the question")
    factually_correct = dspy.OutputField(desc="Is the answer factually correct based on the context?", prefix="Factual[Yes/No]:")

judge = dspy.ChainOfThought(FactJudge)

def factuality_metric(example, pred):
    factual = judge(context=example.context, question=example.question, answer=pred.answer)
    return int(factual=="Yes")
```

## DSPy Evaluation

```python
from dspy.evaluate import Evaluate

evaluate_program = Evaluate(devset=devset, metric=your_defined_metric, num_threads=NUM_THREADS, display_progress=True, display_table=num_rows_to_display)

evaluate_program(your_dspy_program)
```

## DSPy Optimizers

### LabeledFewShot

```python
from dspy.teleprompt import LabeledFewShot

labeled_fewshot_optimizer = LabeledFewShot(k=8)
your_dspy_program_compiled = labeled_fewshot_optimizer.compile(student = your_dspy_program, trainset=trainset)
```

### BootstrapFewShot

```python
from dspy.teleprompt import BootstrapFewShot

fewshot_optimizer = BootstrapFewShot(metric=your_defined_metric, max_bootstrapped_demos=4, max_labeled_demos=16, max_rounds=1, max_errors=5)

your_dspy_program_compiled = fewshot_optimizer.compile(student = your_dspy_program, trainset=trainset)
```

#### Using another LM for compilation, specifying in teacher_settings

```python
from dspy.teleprompt import BootstrapFewShot

fewshot_optimizer = BootstrapFewShot(metric=your_defined_metric, max_bootstrapped_demos=4, max_labeled_demos=16, max_rounds=1, max_errors=5, teacher_settings=dict(lm=gpt4))

your_dspy_program_compiled = fewshot_optimizer.compile(student = your_dspy_program, trainset=trainset)
```

#### Compiling a compiled program - bootstrapping a bootstrapped program

```python
your_dspy_program_compiledx2 = teleprompter.compile(
    your_dspy_program,
    teacher=your_dspy_program_compiled,
    trainset=trainset,
)
```

#### Saving/loading a compiled program

```python
save_path = './v1.json'
your_dspy_program_compiledx2.save(save_path)
```

```python
loaded_program = YourProgramClass()
loaded_program.load(path=save_path)
```

### BootstrapFewShotWithRandomSearch

Detailed documentation on BootstrapFewShotWithRandomSearch can be found [here](deep-dive/optimizers/bootstrap-fewshot.md).

```python
from dspy.teleprompt import BootstrapFewShotWithRandomSearch

fewshot_optimizer = BootstrapFewShotWithRandomSearch(metric=your_defined_metric, max_bootstrapped_demos=2, num_candidate_programs=8, num_threads=NUM_THREADS)

your_dspy_program_compiled = fewshot_optimizer.compile(student = your_dspy_program, trainset=trainset, valset=devset)

```

Other custom configurations are similar to customizing the `BootstrapFewShot` optimizer.

### Ensemble

```python
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from dspy.teleprompt.ensemble import Ensemble

fewshot_optimizer = BootstrapFewShotWithRandomSearch(metric=your_defined_metric, max_bootstrapped_demos=2, num_candidate_programs=8, num_threads=NUM_THREADS)
your_dspy_program_compiled = fewshot_optimizer.compile(student = your_dspy_program, trainset=trainset, valset=devset)

ensemble_optimizer = Ensemble(reduce_fn=dspy.majority)
programs = [x[-1] for x in your_dspy_program_compiled.candidate_programs]
your_dspy_program_compiled_ensemble = ensemble_optimizer.compile(programs[:3])
```

### BootstrapFinetune

```python
from dspy.teleprompt import BootstrapFewShotWithRandomSearch, BootstrapFinetune

#Compile program on current dspy.settings.lm
fewshot_optimizer = BootstrapFewShotWithRandomSearch(metric=your_defined_metric, max_bootstrapped_demos=2, num_threads=NUM_THREADS)
your_dspy_program_compiled = tp.compile(your_dspy_program, trainset=trainset[:some_num], valset=trainset[some_num:])

#Configure model to finetune
config = dict(target=model_to_finetune, epochs=2, bf16=True, bsize=6, accumsteps=2, lr=5e-5)

#Compile program on BootstrapFinetune
finetune_optimizer = BootstrapFinetune(metric=your_defined_metric)
finetune_program = finetune_optimizer.compile(your_dspy_program, trainset=some_new_dataset_for_finetuning_model, **config)

finetune_program = your_dspy_program

#Load program and activate model's parameters in program before evaluation
ckpt_path = "saved_checkpoint_path_from_finetuning"
LM = dspy.HFModel(checkpoint=ckpt_path, model=model_to_finetune)

for p in finetune_program.predictors():
    p.lm = LM
    p.activated = False
```

### COPRO

Detailed documentation on COPRO can be found [here](deep-dive/optimizers/copro.md).

```python
from dspy.teleprompt import COPRO

eval_kwargs = dict(num_threads=16, display_progress=True, display_table=0)

copro_teleprompter = COPRO(prompt_model=model_to_generate_prompts, metric=your_defined_metric, breadth=num_new_prompts_generated, depth=times_to_generate_prompts, init_temperature=prompt_generation_temperature, verbose=False)

compiled_program_optimized_signature = copro_teleprompter.compile(your_dspy_program, trainset=trainset, eval_kwargs=eval_kwargs)
```

### MIPRO

```python
from dspy.teleprompt import MIPRO

teleprompter = MIPRO(prompt_model=model_to_generate_prompts, task_model=model_that_solves_task, metric=your_defined_metric, num_candidates=num_new_prompts_generated, init_temperature=prompt_generation_temperature)

kwargs = dict(num_threads=NUM_THREADS, display_progress=True, display_table=0)

compiled_program_optimized_bayesian_signature = teleprompter.compile(your_dspy_program, trainset=trainset, num_trials=100, max_bootstrapped_demos=3, max_labeled_demos=5, eval_kwargs=kwargs)
```

### MIPROv2

Note: detailed documentation can be found [here](deep-dive/optimizers/miprov2.md). `MIPROv2` is the latest extension of `MIPRO` which includes updates such as (1) improvements to instruction proposal and (2) more efficient search with minibatching.

#### Optimizing with MIPROv2

This shows how to perform an easy out-of-the box run with `auto=light`, which configures many hyperparameters for you and performs a light optimization run. You can alternatively set `auto=medium` or `auto=heavy` to perform longer optimization runs. The more detailed `MIPROv2` documentation [here](deep-dive/optimizers/miprov2.md) also provides more information about how to set hyperparameters by hand.

```python
# Import the optimizer
from dspy.teleprompt import MIPROv2

# Initialize optimizer
teleprompter = MIPROv2(
    metric=gsm8k_metric,
    auto="light", # Can choose between light, medium, and heavy optimization runs
)

# Optimize program
print(f"Optimizing program with MIPRO...")
optimized_program = teleprompter.compile(
    program.deepcopy(),
    trainset=trainset,
    max_bootstrapped_demos=3,
    max_labeled_demos=4,
    requires_permission_to_run=False,
)

# Save optimize program for future use
optimized_program.save(f"mipro_optimized")

# Evaluate optimized program
print(f"Evaluate optimized program...")
evaluate(optimized_program, devset=devset[:])
```

#### Optimizing instructions only with MIPROv2 (0-Shot)

```python
# Import the optimizer
from dspy.teleprompt import MIPROv2

# Initialize optimizer
teleprompter = MIPROv2(
    metric=gsm8k_metric,
    auto="light", # Can choose between light, medium, and heavy optimization runs
)

# Optimize program
print(f"Optimizing program with MIPRO...")
optimized_program = teleprompter.compile(
    program.deepcopy(),
    trainset=trainset,
    max_bootstrapped_demos=0,
    max_labeled_demos=0,
    requires_permission_to_run=False,
)

# Save optimize program for future use
optimized_program.save(f"mipro_optimized")

# Evaluate optimized program
print(f"Evaluate optimized program...")
evaluate(optimized_program, devset=devset[:])
```

### Signature Optimizer with Types

```python
from dspy.teleprompt.signature_opt_typed import optimize_signature
from dspy.evaluate.metrics import answer_exact_match
from dspy.functional import TypedChainOfThought

compiled_program = optimize_signature(
    student=TypedChainOfThought("question -> answer"),
    evaluator=Evaluate(devset=devset, metric=answer_exact_match, num_threads=10, display_progress=True),
    n_iterations=50,
).program
```

### KNNFewShot

```python
from sentence_transformers import SentenceTransformer
from dspy import Embedder
from dspy.teleprompt import KNNFewShot
from dspy import ChainOfThought

knn_optimizer = KNNFewShot(k=3, trainset=trainset, vectorizer=Embedder(SentenceTransformer("all-MiniLM-L6-v2").encode))

qa_compiled = knn_optimizer.compile(student=ChainOfThought("question -> answer"))
```

### BootstrapFewShotWithOptuna

```python
from dspy.teleprompt import BootstrapFewShotWithOptuna

fewshot_optuna_optimizer = BootstrapFewShotWithOptuna(metric=your_defined_metric, max_bootstrapped_demos=2, num_candidate_programs=8, num_threads=NUM_THREADS)

your_dspy_program_compiled = fewshot_optuna_optimizer.compile(student=your_dspy_program, trainset=trainset, valset=devset)
```

Other custom configurations are similar to customizing the `dspy.BootstrapFewShot` optimizer.

## DSPy `Refine` and `BestofN`

>`dspy.Suggest` and `dspy.Assert` are replaced by `dspy.Refine` and `dspy.BestofN` in DSPy 2.6.

### BestofN

Runs a module up to `N` times with different temperatures and returns the best prediction, as defined by the `reward_fn`, or the first prediction that passes the `threshold`.

```python
import dspy

qa = dspy.ChainOfThought("question -> answer")
def one_word_answer(args, pred):
    return 1.0 if len(pred.answer) == 1 else 0.0
best_of_3 = dspy.BestOfN(module=qa, N=3, reward_fn=one_word_answer, threshold=1.0)
best_of_3(question="What is the capital of Belgium?").answer
# Brussels
```

### Refine

Refines a module by running it up to `N` times with different temperatures and returns the best prediction, as defined by the `reward_fn`, or the first prediction that passes the `threshold`. After each attempt (except the final one), `Refine` automatically generates detailed feedback about the module's performance and uses this feedback as hints for subsequent runs, creating an iterative refinement process.

```python
import dspy

qa = dspy.ChainOfThought("question -> answer")
def one_word_answer(args, pred):
    return 1.0 if len(pred.answer) == 1 else 0.0
best_of_3 = dspy.Refine(module=qa, N=3, reward_fn=one_word_answer, threshold=1.0)
best_of_3(question="What is the capital of Belgium?").answer
# Brussels
```

#### Error Handling

By default, `Refine` will try to run the module up to N times until the threshold is met. If the module encounters an error, it will keep going up to N failed attempts. You can change this behavior by setting `fail_count` to a smaller number than `N`.

```python
refine = dspy.Refine(module=qa, N=3, reward_fn=one_word_answer, threshold=1.0, fail_count=1)
...
refine(question="What is the capital of Belgium?")
# If we encounter just one failed attempt, the module will raise an error.
```

If you want to run the module up to N times without any error handling, you can set `fail_count` to `N`. This is the default behavior.

```python
refine = dspy.Refine(module=qa, N=3, reward_fn=one_word_answer, threshold=1.0, fail_count=3)
...
refine(question="What is the capital of Belgium?")
```


## End of cheatsheet.md

---


## File: index_.md

---
template: home.html
hide:
  - navigation
  - toc
---


## End of index_.md

---


## File: roadmap.md

---
draft: true
---


# Roadmap Sketch: DSPy 2.5+

It’s been a year since DSPy evolved out of Demonstrate–Search–Predict (DSP), whose research started at Stanford NLP all the way back in February 2022. Thanks to 200 wonderful contributors, DSPy has introduced tens of thousands of people to building modular LM programs and optimizing their prompts and weights automatically. In this time, DSPy has grown to 160,000 monthly downloads and 16,000 stars on GitHub, becoming synonymous with prompt optimization in many circles and inspiring at least a half-dozen cool new libraries.

This document is an initial sketch of DSPy’s public roadmap for the next few weeks and months, as we work on DSPy 2.5 and plan for DSPy 3.0. Suggestions and open-source contributors are more than welcome: just open an issue or submit a pull request regarding the roadmap.



## Technical Objectives

The thesis of DSPy is that for LMs to be useful, we have to shift from ad-hoc prompting to new notions of programming LMs. Instead of relying on LMs gaining much more general or more compositional capabilities, we need to enable developers to iteratively explore their problems and build modular software that invokes LMs for well-scoped tasks. We need to enable that through modules and optimizers that isolate how they decompose their problems and describe their system's objectives from how their LMs are invoked or fine-tuned to maximize their objectives. DSPy's goal has been to develop (and to build the community and shared infrastructure for the collective development of) the abstractions, programming patterns, and optimizers toward this thesis.

To a first approximation, DSPy’s current user-facing language has the minimum number of appropriate abstractions that address the goals above: declarative signatures, define-by-run modules, and optimizers that can be composed quite powerfully. But there are several things we need to do better to realize our goals. The upcoming DSPy releases will have the following objectives.

1. Polishing the core functionality.
2. Developing more accurate, lower-cost optimizers.
3. Building end-to-end tutorials from DSPy’s ML workflow to deployment.
4. Shifting towards more interactive optimization & tracking.



## Team & Organization

DSPy is fairly unusual in its technical objectives, contributors, and audience. Though DSPy takes inspiration from PyTorch, a library for building and optimizing DNNs, there is one major difference: PyTorch was introduced well after DNNs were mature ML concepts, but DSPy seeks to establish and advance core LM Programs research: the framework is propelled by constant academic research from programming abstractions (like the original **Demonstrate–Search–Predict** concepts, DSPy **Signatures**, or **LM Assertions**) to NLP systems (like **STORM**, **PATH**, and **IReRa**) to prompt optimizers (like **MIPRO**) and RL (like **BetterTogether**), among many other related directions.

This research all composes into a concrete, practical library, thanks to dozens of industry contributors, many of whom are deploying apps in production using DSPy. Because of this, DSPy reaches not only of grad students and ML engineers, but also many non-ML engineers, from early adopter SWEs to hobbyists exploring new ways of using LMs. The following team, with help from many folks in the OSS community, is working towards the objectives in this Roadmap.

**Project Lead:** Omar Khattab (Stanford & Databricks)

**Project Mentors:** Chris Potts (Stanford), Matei Zaharia (UC Berkeley & Databricks), Heather Miller (CMU & Two Sigma)

**Core Library:** Arnav Singhvi (Databricks & Stanford), Herumb Shandilya (Stanford), Hanna Moazam (Databricks), Sri Vardhamanan (Dashworks), Cyrus Nouroozi (Zenbase), Amir Mehr (Zenbase), Kyle Caverly (Modular), with special thanks to Keshav Santhanam (Stanford), Thomas Ahle (Normal Computing), Connor Shorten (Weaviate)

**Prompt Optimization:** Krista Opsahl-Ong (Stanford), Michael Ryan (Stanford), Josh Purtell (Basis), with special thanks to Eric Zhang (Stanford)

**Finetuning & RL:** Dilara Soylu (Stanford), Isaac Miller (Anyscale), Karel D'Oosterlinck (Ghent), with special thanks to Paridhi Masehswari (Stanford)

**PL Abstractions:** Shangyin Tan (UC Berkeley), Manish Shetty (UC Berkeley), Peter Zhong (CMU)

**Applications:** Jasper Xian (Waterloo), Saron Samuel (Stanford), Alberto Mancarella (Stanford), Faraz Khoubsirat (Waterloo), Saiful Haq (IIT-B), Ashutosh Sharma (UIUC)



## 1) Polishing the core functionality.

Over the next month, polishing is the main objective and likely the one to have the highest ROI on the experience of the average user. Conceptually, DSPy has an extremely small core. It’s nothing but (1) LMs, (2) Signatures & Modules, (3) Optimizers, and (4) Assertions. These concepts and their implementations evolved organically over the past couple of years. We are working now to consolidate what we’ve learned and refactor internally so that things “just work” out of the box for new users, who may not know all the tips-and-tricks just yet.

More concretely:

1. We want to increase the quality of zero-shot, off-the-shelf DSPy programs, i.e. those not yet compiled on custom data.
2. Wherever possible, DSPy should delegate lower-level internal complexity (like managing LMs and structured generation) to emerging lower-level libraries. When required, we may fork smaller libraries out of DSPy to support infrastructure pieces as their own projects.
3. DSPy should internally be more modular and we need higher compatibility between internal components. Specifically, we need more deeper and more native investment in (i) typed multi-field constraints, (ii) assertions, (iii) observability and experimental tracking, (iv) deployment of artifacts and related concerns like streaming and async, and (v) fine-tuning and serving open models.


### On LMs

As of DSPy 2.4, the library has approximately 20,000 lines of code and roughly another 10,000 lines of code for tests, examples, and documentation. Some of these are clearly necessary (e.g., DSPy optimizers) but others exist only because the LM space lacks the building blocks we need under the hood. Luckily, for LM interfaces, a very strong library now exists: LiteLLM, a library that unifies interfaces to various LM and embedding providers. We expect to reduce around 6000 LoCs of support for custom LMs and retrieval models by shifting a lot of that to LiteLLM.

Objectives in this space include improved caching, saving/loading of LMs, support for streaming and async LM requests. Work here is currently led by Hanna Moazam and Sri Vardhamanan, building on a foundation by Cyrus Nouroozi, Amir Mehr, Kyle Caverly, and others.


### On Signatures & Modules

Traditionally, LMs offer text-in-text-out interfaces. Toward modular programming, DSPy introduced signatures for the first time (as DSP Templates in Jan 2023) as a way to structure the inputs and outputs of LM interactions. Standard prompts conflate interface (“what should the LM do?”) with implementation (“how do we tell it to do that?”). DSPy signatures isolate the former so we can infer and learn the latter from data — in the context of a bigger program. Today in the LM landscape, notions of "structured outputs" have evolved dramatically, thanks to constrained decoding and other improvements, and have become mainstream. What may be called "structured inputs" remains is yet to become mainstream outside of DSPy, but is as crucial.

Objectives in this space include refining the abstractions and implementations first-class notion of LM Adapters in DSPy, as translators that sits between signatures and LM interfaces. While Optimizers adjust prompts through interactions with a user-supplied metric and data, Adapters are more concerned with building up interactions with LMs to account for, e.g. (i) non-plaintext LM interfaces like chat APIs, structured outputs, function calling, and multi-modal APIs, (ii) languages beyond English or other forms of higher-level specialization. This has been explored in DSPy on and off in various forms, but we have started working on more fundamental approaches to this problem that will offer tangible improvements to most use-cases. Work here is currently led by Omar Khattab.


### On Finetuning & Serving

In February 2023, DSPy introduced the notion of compiling to optimize the weights of an LM program. (To understand just how long ago that was in AI terms, this was before the Alpaca training project at Stanford had even started and a month before the first GPT-4 was released.) Since then, we have shown in October 2023 and, much more expansively, in July 2024, that the fine-tuning flavor of DSPy can deliver large gains for small LMs, especially when composed with prompt optimization.

Overall, though, most DSPy users in practice explore prompt optimization and not weight optimization and most of our examples do the same. The primary reason for a lot of this is infrastructure. Fine-tuning in the DSPy flavor is more than just training a model: ultimately, we need to bootstrap training data for several different modules in a program, train multiple models and handle model selection, and then load and plug in those models into the program's modules. Doing this robustly at the level of abstraction DSPy offers requires a level of resource management that is not generally supported by external existing tools. Major efforts in this regard are currently led by Dilara Soylu and Isaac Miller.


### On Optimizers & Assertions

This is a naturally major direction in the course of polishing. We will share more thoughts here after making more progress on the three angles above.



## 2) Developing more accurate, lower-cost optimizers.

A very large fraction of the research in DSPy focuses on optimizing the prompts and the weights of LM programs. In December 2022, we introduced the algorithm and abstractions behind BootstrapFewShot (as Demonstrate in DSP) and several of its variants. In February 2023, we introduced the core version of what later became BootstrapFinetune. In August 2023, we introduced new variations of both of these. In December 2023, we introduced the first couple of instruction optimizers into DSPy, CA-OPRO and early versions of MIPRO. These were again upgraded in March 2024. Fast forward to June and July 2024, we released MIPROv2 for prompt optimization and BetterTogether for fine-tuning the weights of LM programs.

We have been working towards a number of stronger optimizers. While we cannot share the internal details of research on new optimizers yet, we can outline the goals. A DSPy optimizer can be characterized via three angles:

1. Quality: How much quality can it deliver from various LMs? How effective does it need the zero-shot program to be in order to work well?
2. Cost: How many labeled (and unlabeled) inputs does it need? How many invocations of the program does it need? How expensive is the resulting optimized program at inference time?
3. Robustness: How well can it generalize to different unseen data points or distributions? How sensitive is it to mistakes of the metric or labels?

Over the next six months, our goal is to dramatically improve each angle of these _when the other two are held constant_.  Concretely, there are three directions here.

- Benchmarking: A key prerequisite here is work on benchmarking. On the team, Michael Ryan and Shangyin Tan are leading these efforts. More soon.

- Quality: The goal here is optimizers that extract, on average, 20% more on representative tasks than MIPROv2 and BetterTogether, under the usual conditions — like a few hundred inputs with labels and a good metric starting from a decent zero-shot program. Various efforts here are led by Dilara Soylu, Michael Ryan, Josh Purtell, Krista Opsahl-Ong, and Isaac Miller.

- Efficiency: The goal here is optimizers that match the current best scores from MIPROv2 and BetterTogether but under 1-2 challenges like: (i) starting from only 10-20 inputs with labels, (ii) starting with a weak zero-shot program that scores 0%, (iii) where significant misalignment exists between train/validation and test, or (iii) where the user supplies no metric but provides a very small number of output judgments.



## 3) Building end-to-end tutorials from DSPy’s ML workflow to deployment.

Using DSPy well for solving a new task is just doing good machine learning with LMs, but teaching this is hard. On the one hand, it's an iterative process: you make some initial choices, which will be sub-optimal, and then you refine them incrementally. It's highly exploratory: it's often the case that no one knows yet how to best solve a problem in a DSPy-esque way. One the other hand, DSPy offers many emerging lessons from several years of building LM systems, in which the design space, the data regime, and many other factors are new both to ML experts and to the very large fraction of users that have no ML experience.

Though current docs do address [a bunch of this](/building-blocks/solving_your_task) in isolated ways, one thing we've learned is that we should separate teaching the core DSPy language (which is ultimately pretty small) from teaching the emerging ML workflow that works well in a DSPy-esque setting. As a natural extension of this, we need to place more emphasis on steps prior and after to the explicit coding in DSPy, from data collection to deployment that serves and monitors the optimized DSPy program in practice. This is just starting but efforts will be ramping up led by Omar Khattab, Isaac Miller, and Herumb Shandilya.


## 4) Shifting towards more interactive optimization & tracking.

Right now, a DSPy user has a few ways to observe and tweak the process of optimization. They can study the prompts before, during, and after optimization methods like `inspect_history`, built-in logging, and/or the metadata returned by optimizers. Similarly, they can rely on `program.save` and `program.load` to potentially adjust the optimized prompts by hand. Alternatively, they can use one of the many powerful observability integrations — like from Phoenix Arize, LangWatch, or Weights & Biases Weave — to observe _in real time_ the process of optimization (e.g., scores, stack traces, successful & failed traces, and candidate prompts). DSPy encourages iterative engineering by adjusting the program, data, or metrics across optimization runs. For example, some optimizers allow “checkpointing” — e.g., if you optimize with BootstrapFewShotWithRandomSearch for 10 iterations then increase to 15 iterations, the first 10 will be loaded from cache.

While these can accomplish a lot of goals, there are two limitations that future versions of DSPy will seek to address.

1. In general, DSPy’s (i) observability, (ii) experimental tracking, (iii) cost management, and (iii) deployment of programs should become first-class concerns via integration with tools like MLFlow. We will share more plans addressing this for DSPy 2.6 in the next 1-2 months.

2. DSPy 3.0 will introduce new optimizers that prioritize ad-hoc, human-in-the-loop feedback. This is perhaps the only substantial paradigm shift we see as necessary in the foreseeable future in DSPy. It involves various research questions at the level of the abstractions, UI/HCI, and ML, so it is a longer-term goal that we will share more about in the next 3-4 month.




## End of roadmap.md

---


## File: index.md

---
sidebar_position: 1
hide:
  - navigation
  - toc

---

![DSPy](static/img/dspy_logo.png){ width="200", align=left }

# _Programming_—not prompting—_LMs_

[![PyPI Downloads](https://static.pepy.tech/badge/dspy/month)](https://pepy.tech/projects/dspy)

DSPy is the framework for _programming—rather than prompting—language models_. It allows you to iterate fast on **building modular AI systems** and offers algorithms for **optimizing their prompts and weights**, whether you're building simple classifiers, sophisticated RAG pipelines, or Agent loops.

DSPy stands for Declarative Self-improving Python. Instead of brittle prompts, you write compositional _Python code_ and use DSPy to **teach your LM to deliver high-quality outputs**. This [lecture](https://www.youtube.com/watch?v=JEMYuzrKLUw) is a good conceptual introduction. Meet the community, seek help, or start contributing via our [GitHub repo](https://github.com/stanfordnlp/dspy) and [Discord server](https://discord.gg/XCGy2WDCQB).


!!! info "Getting Started I: Install DSPy and set up your LM"

    ```bash
    > pip install -U dspy
    ```

    === "OpenAI"
        You can authenticate by setting the `OPENAI_API_KEY` env variable or passing `api_key` below.

        ```python linenums="1"
        import dspy
        lm = dspy.LM('openai/gpt-4o-mini', api_key='YOUR_OPENAI_API_KEY')
        dspy.configure(lm=lm)
        ```

    === "Anthropic"
        You can authenticate by setting the ANTHROPIC_API_KEY env variable or passing `api_key` below.

        ```python linenums="1"
        import dspy
        lm = dspy.LM('anthropic/claude-3-opus-20240229', api_key='YOUR_ANTHROPIC_API_KEY')
        dspy.configure(lm=lm)
        ```

    === "Databricks"
        If you're on the Databricks platform, authentication is automatic via their SDK. If not, you can set the env variables `DATABRICKS_API_KEY` and `DATABRICKS_API_BASE`, or pass `api_key` and `api_base` below.

        ```python linenums="1"
        import dspy
        lm = dspy.LM('databricks/databricks-meta-llama-3-1-70b-instruct')
        dspy.configure(lm=lm)
        ```

    === "Local LMs on your laptop"
          First, install [Ollama](https://github.com/ollama/ollama) and launch its server with your LM.

          ```bash
          > curl -fsSL https://ollama.ai/install.sh | sh
          > ollama run llama3.2:1b
          ```

          Then, connect to it from your DSPy code.

        ```python linenums="1"
        import dspy
        lm = dspy.LM('ollama_chat/llama3.2', api_base='http://localhost:11434', api_key='')
        dspy.configure(lm=lm)
        ```

    === "Local LMs on a GPU server"
          First, install [SGLang](https://sgl-project.github.io/start/install.html) and launch its server with your LM.

          ```bash
          > pip install "sglang[all]"
          > pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/ 

          > CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server --port 7501 --model-path meta-llama/Llama-3.1-8B-Instruct
          ```
        
        If you don't have access from Meta to download `meta-llama/Llama-3.1-8B-Instruct`, use `Qwen/Qwen2.5-7B-Instruct` for example.

        Next, connect to your local LM from your DSPy code as an `OpenAI`-compatible endpoint.

          ```python linenums="1"
          lm = dspy.LM("openai/meta-llama/Llama-3.1-8B-Instruct",
                       api_base="http://localhost:7501/v1",  # ensure this points to your port
                       api_key="local", model_type='chat')
          dspy.configure(lm=lm)
          ```

    === "Other providers"
        In DSPy, you can use any of the dozens of [LLM providers supported by LiteLLM](https://docs.litellm.ai/docs/providers). Simply follow their instructions for which `{PROVIDER}_API_KEY` to set and how to write pass the `{provider_name}/{model_name}` to the constructor.

        Some examples:

        - `anyscale/mistralai/Mistral-7B-Instruct-v0.1`, with `ANYSCALE_API_KEY`
        - `together_ai/togethercomputer/llama-2-70b-chat`, with `TOGETHERAI_API_KEY`
        - `sagemaker/<your-endpoint-name>`, with `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `AWS_REGION_NAME`
        - `azure/<your_deployment_name>`, with `AZURE_API_KEY`, `AZURE_API_BASE`, `AZURE_API_VERSION`, and the optional `AZURE_AD_TOKEN` and `AZURE_API_TYPE`

        
        If your provider offers an OpenAI-compatible endpoint, just add an `openai/` prefix to your full model name.

        ```python linenums="1"
        import dspy
        lm = dspy.LM('openai/your-model-name', api_key='PROVIDER_API_KEY', api_base='YOUR_PROVIDER_URL')
        dspy.configure(lm=lm)
        ```

??? "Calling the LM directly."

     Idiomatic DSPy involves using _modules_, which we define in the rest of this page. However, it's still easy to call the `lm` you configured above directly. This gives you a unified API and lets you benefit from utilities like automatic caching.

     ```python linenums="1"       
     lm("Say this is a test!", temperature=0.7)  # => ['This is a test!']
     lm(messages=[{"role": "user", "content": "Say this is a test!"}])  # => ['This is a test!']
     ``` 


## 1) **Modules** help you describe AI behavior as _code_, not strings.

To build reliable AI systems, you must iterate fast. But maintaining prompts makes that hard: it forces you to tinker with strings or data _every time you change your LM, metrics, or pipeline_. Having built over a dozen best-in-class compound LM systems since 2020, we learned this the hard way—and so built DSPy to decouple AI system design from messy incidental choices about specific LMs or prompting strategies.

DSPy shifts your focus from tinkering with prompt strings to **programming with structured and declarative natural-language modules**. For every AI component in your system, you specify input/output behavior as a _signature_ and select a _module_ to assign a strategy for invoking your LM. DSPy expands your signatures into prompts and parses your typed outputs, so you can compose different modules together into ergonomic, portable, and optimizable AI systems.


!!! info "Getting Started II: Build DSPy modules for various tasks"
    Try the examples below after configuring your `lm` above. Adjust the fields to explore what tasks your LM can do well out of the box. Each tab below sets up a DSPy module, like `dspy.Predict`, `dspy.ChainOfThought`, or `dspy.ReAct`, with a task-specific _signature_. For example, `question -> answer: float` tells the module to take a question and to produce a `float` answer.

    === "Math"

        ```python linenums="1"
        math = dspy.ChainOfThought("question -> answer: float")
        math(question="Two dice are tossed. What is the probability that the sum equals two?")
        ```
        
        **Possible Output:**
        ```text
        Prediction(
            reasoning='When two dice are tossed, each die has 6 faces, resulting in a total of 6 x 6 = 36 possible outcomes. The sum of the numbers on the two dice equals two only when both dice show a 1. This is just one specific outcome: (1, 1). Therefore, there is only 1 favorable outcome. The probability of the sum being two is the number of favorable outcomes divided by the total number of possible outcomes, which is 1/36.',
            answer=0.0277776
        )
        ```

    === "RAG"

        ```python linenums="1"       
        def search_wikipedia(query: str) -> list[str]:
            results = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')(query, k=3)
            return [x['text'] for x in results]
        
        rag = dspy.ChainOfThought('context, question -> response')

        question = "What's the name of the castle that David Gregory inherited?"
        rag(context=search_wikipedia(question), question=question)
        ```
        
        **Possible Output:**
        ```text
        Prediction(
            reasoning='The context provides information about David Gregory, a Scottish physician and inventor. It specifically mentions that he inherited Kinnairdy Castle in 1664. This detail directly answers the question about the name of the castle that David Gregory inherited.',
            response='Kinnairdy Castle'
        )
        ```

    === "Classification"

        ```python linenums="1"
        from typing import Literal

        class Classify(dspy.Signature):
            """Classify sentiment of a given sentence."""
            
            sentence: str = dspy.InputField()
            sentiment: Literal['positive', 'negative', 'neutral'] = dspy.OutputField()
            confidence: float = dspy.OutputField()

        classify = dspy.Predict(Classify)
        classify(sentence="This book was super fun to read, though not the last chapter.")
        ```
        
        **Possible Output:**

        ```text
        Prediction(
            sentiment='positive',
            confidence=0.75
        )
        ```

    === "Information Extraction"

        ```python linenums="1"        
        class ExtractInfo(dspy.Signature):
            """Extract structured information from text."""
            
            text: str = dspy.InputField()
            title: str = dspy.OutputField()
            headings: list[str] = dspy.OutputField()
            entities: list[dict[str, str]] = dspy.OutputField(desc="a list of entities and their metadata")
        
        module = dspy.Predict(ExtractInfo)

        text = "Apple Inc. announced its latest iPhone 14 today." \
            "The CEO, Tim Cook, highlighted its new features in a press release."
        response = module(text=text)

        print(response.title)
        print(response.headings)
        print(response.entities)
        ```
        
        **Possible Output:**
        ```text
        Apple Inc. Announces iPhone 14
        ['Introduction', "CEO's Statement", 'New Features']
        [{'name': 'Apple Inc.', 'type': 'Organization'}, {'name': 'iPhone 14', 'type': 'Product'}, {'name': 'Tim Cook', 'type': 'Person'}]
        ```

    === "Agents"

        ```python linenums="1"       
        def evaluate_math(expression: str):
            return dspy.PythonInterpreter({}).execute(expression)

        def search_wikipedia(query: str):
            results = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')(query, k=3)
            return [x['text'] for x in results]

        react = dspy.ReAct("question -> answer: float", tools=[evaluate_math, search_wikipedia])

        pred = react(question="What is 9362158 divided by the year of birth of David Gregory of Kinnairdy castle?")
        print(pred.answer)
        ```
        
        **Possible Output:**

        ```text
        5761.328
        ```
    
    === "Multi-Stage Pipelines"

        ```python linenums="1"       
        class Outline(dspy.Signature):
            """Outline a thorough overview of a topic."""
            
            topic: str = dspy.InputField()
            title: str = dspy.OutputField()
            sections: list[str] = dspy.OutputField()
            section_subheadings: dict[str, list[str]] = dspy.OutputField(desc="mapping from section headings to subheadings")

        class DraftSection(dspy.Signature):
            """Draft a top-level section of an article."""
            
            topic: str = dspy.InputField()
            section_heading: str = dspy.InputField()
            section_subheadings: list[str] = dspy.InputField()
            content: str = dspy.OutputField(desc="markdown-formatted section")

        class DraftArticle(dspy.Module):
            def __init__(self):
                self.build_outline = dspy.ChainOfThought(Outline)
                self.draft_section = dspy.ChainOfThought(DraftSection)

            def forward(self, topic):
                outline = self.build_outline(topic=topic)
                sections = []
                for heading, subheadings in outline.section_subheadings.items():
                    section, subheadings = f"## {heading}", [f"### {subheading}" for subheading in subheadings]
                    section = self.draft_section(topic=outline.title, section_heading=section, section_subheadings=subheadings)
                    sections.append(section.content)
                return dspy.Prediction(title=outline.title, sections=sections)

        draft_article = DraftArticle()
        article = draft_article(topic="World Cup 2002")
        ```
        
        **Possible Output:**

        A 1500-word article on the topic, e.g.

        ```text
        ## Qualification Process

        The qualification process for the 2002 FIFA World Cup involved a series of..... [shortened here for presentation].

        ### UEFA Qualifiers

        The UEFA qualifiers involved 50 teams competing for 13..... [shortened here for presentation].

        .... [rest of the article]
        ```

        Note that DSPy makes it straightforward to optimize multi-stage modules like this. As long as you can evaluate the _final_ output of the system, every DSPy optimizer can tune all of the intermediate modules.

??? "Using DSPy in practice: from quick scripting to building sophisticated systems."

    Standard prompts conflate interface (“what should the LM do?”) with implementation (“how do we tell it to do that?”). DSPy isolates the former as _signatures_ so we can infer the latter or learn it from data — in the context of a bigger program.
    
    Even before you start using optimizers, DSPy's modules allow you to script effective LM systems as ergonomic, portable _code_. Across many tasks and LMs, we maintain _signature test suites_ that assess the reliability of the built-in DSPy adapters. Adapters are the components that map signatures to prompts prior to optimization. If you find a task where a simple prompt consistently outperforms idiomatic DSPy for your LM, consider that a bug and [file an issue](https://github.com/stanfordnlp/dspy/issues). We'll use this to improve the built-in adapters.


## 2) **Optimizers** tune the prompts and weights of your AI modules.

DSPy provides you with the tools to compile high-level code with natural language annotations into the low-level computations, prompts, or weight updates that align your LM with your program’s structure and metrics. If you change your code or your metrics, you can simply re-compile accordingly.

Given a few tens or hundreds of representative _inputs_ of your task and a _metric_ that can measure the quality of your system's outputs, you can use a DSPy optimizer. Different optimizers in DSPy work by **synthesizing good few-shot examples** for every module, like `dspy.BootstrapRS`,<sup>[1](https://arxiv.org/abs/2310.03714)</sup> **proposing and intelligently exploring better natural-language instructions** for every prompt, like `dspy.MIPROv2`,<sup>[2](https://arxiv.org/abs/2406.11695)</sup> and **building datasets for your modules and using them to finetune the LM weights** in your system, like `dspy.BootstrapFinetune`.<sup>[3](https://arxiv.org/abs/2407.10930)</sup>


!!! info "Getting Started III: Optimizing the LM prompts or weights in DSPy programs"
    A typical simple optimization run costs on the order of $2 USD and takes around 20 minutes, but be careful when running optimizers with very large LMs or very large datasets.
    Optimization can cost as little as a few cents or up to tens of dollars, depending on your LM, dataset, and configuration.
    
    === "Optimizing prompts for a ReAct agent"
        This is a minimal but fully runnable example of setting up a `dspy.ReAct` agent that answers questions via
        search from Wikipedia and then optimizing it using `dspy.MIPROv2` in the cheap `light` mode on 500
        question-answer pairs sampled from the `HotPotQA` dataset.

        ```python linenums="1"
        import dspy
        from dspy.datasets import HotPotQA

        dspy.configure(lm=dspy.LM('openai/gpt-4o-mini'))

        def search_wikipedia(query: str) -> list[str]:
            results = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')(query, k=3)
            return [x['text'] for x in results]

        trainset = [x.with_inputs('question') for x in HotPotQA(train_seed=2024, train_size=500).train]
        react = dspy.ReAct("question -> answer", tools=[search_wikipedia])

        tp = dspy.MIPROv2(metric=dspy.evaluate.answer_exact_match, auto="light", num_threads=24)
        optimized_react = tp.compile(react, trainset=trainset)
        ```

        An informal run like this raises ReAct's score from 24% to 51%, by teaching `gpt-4o-mini` more about the specifics of the task.

    === "Optimizing prompts for RAG"
        Given a retrieval index to `search`, your favorite `dspy.LM`, and a small `trainset` of questions and ground-truth responses, the following code snippet can optimize your RAG system with long outputs against the built-in `SemanticF1` metric, which is implemented as a DSPy module.

        ```python linenums="1"
        class RAG(dspy.Module):
            def __init__(self, num_docs=5):
                self.num_docs = num_docs
                self.respond = dspy.ChainOfThought('context, question -> response')

            def forward(self, question):
                context = search(question, k=self.num_docs)   # defined in tutorial linked below
                return self.respond(context=context, question=question)

        tp = dspy.MIPROv2(metric=dspy.evaluate.SemanticF1(decompositional=True), auto="medium", num_threads=24)
        optimized_rag = tp.compile(RAG(), trainset=trainset, max_bootstrapped_demos=2, max_labeled_demos=2)
        ```

        For a complete RAG example that you can run, start this [tutorial](/tutorials/rag/). It improves the quality of a RAG system over a subset of StackExchange communities by 10% relative gain.

    === "Optimizing weights for Classification"
        This is a minimal but fully runnable example of setting up a `dspy.ChainOfThought` module that classifies
        short texts into one of 77 banking labels and then using `dspy.BootstrapFinetune` with 2000 text-label pairs
        from the `Banking77` to finetune the weights of GPT-4o-mini for this task. We use the variant
        `dspy.ChainOfThoughtWithHint`, which takes an optional `hint` at bootstrapping time, to maximize the utility of
        the training data. Naturally, hints are not available at test time.

        <details><summary>Click to show dataset setup code.</summary>

        ```python linenums="1"
        import random
        from typing import Literal
        from dspy.datasets import DataLoader
        from datasets import load_dataset

        # Load the Banking77 dataset.
        CLASSES = load_dataset("PolyAI/banking77", split="train", trust_remote_code=True).features['label'].names
        kwargs = dict(fields=("text", "label"), input_keys=("text",), split="train", trust_remote_code=True)

        # Load the first 2000 examples from the dataset, and assign a hint to each *training* example.
        trainset = [
            dspy.Example(x, hint=CLASSES[x.label], label=CLASSES[x.label]).with_inputs("text", "hint")
            for x in DataLoader().from_huggingface(dataset_name="PolyAI/banking77", **kwargs)[:2000]
        ]
        random.Random(0).shuffle(trainset)
        ```
        </details>

        ```python linenums="1"
        import dspy
        dspy.configure(lm=dspy.LM('openai/gpt-4o-mini-2024-07-18'))
        
        # Define the DSPy module for classification. It will use the hint at training time, if available.
        signature = dspy.Signature("text -> label").with_updated_fields('label', type_=Literal[tuple(CLASSES)])
        classify = dspy.ChainOfThoughtWithHint(signature)

        # Optimize via BootstrapFinetune.
        optimizer = dspy.BootstrapFinetune(metric=(lambda x, y, trace=None: x.label == y.label), num_threads=24)
        optimized = optimizer.compile(classify, trainset=trainset)

        optimized(text="What does a pending cash withdrawal mean?")
        ```

        **Possible Output (from the last line):**
        ```text
        Prediction(
            reasoning='A pending cash withdrawal indicates that a request to withdraw cash has been initiated but has not yet been completed or processed. This status means that the transaction is still in progress and the funds have not yet been deducted from the account or made available to the user.',
            label='pending_cash_withdrawal'
        )
        ```

        An informal run similar to this on DSPy 2.5.29 raises GPT-4o-mini's score 66% to 87%.


??? "What's an example of a DSPy optimizer? How do different optimizers work?"

    Take the `dspy.MIPROv2` optimizer as an example. First, MIPRO starts with the **bootstrapping stage**. It takes your program, which may be unoptimized at this point, and runs it many times across different inputs to collect traces of input/output behavior for each one of your modules. It filters these traces to keep only those that appear in trajectories scored highly by your metric. Second, MIPRO enters its **grounded proposal stage**. It previews your DSPy program's code, your data, and traces from running your program, and uses them to draft many potential instructions for every prompt in your program. Third, MIPRO launches the **discrete search stage**. It samples mini-batches from your training set, proposes a combination of instructions and traces to use for constructing every prompt in the pipeline, and evaluates the candidate program on the mini-batch. Using the resulting score, MIPRO updates a surrogate model that helps the proposals get better over time.

    One thing that makes DSPy optimizers so powerful is that they can be composed. You can run `dspy.MIPROv2` and use the produced program as an input to `dspy.MIPROv2` again or, say, to `dspy.BootstrapFinetune` to get better results. This is partly the essence of `dspy.BetterTogether`. Alternatively, you can run the optimizer and then extract the top-5 candidate programs and build a `dspy.Ensemble` of them. This allows you to scale _inference-time compute_ (e.g., ensembles) as well as DSPy's unique _pre-inference time compute_ (i.e., optimization budget) in highly systematic ways.



<!-- Future:
BootstrapRS or MIPRO on ??? with a local SGLang LM
BootstrapFS on MATH with a tiny LM like Llama-3.2 with Ollama (maybe with a big teacher) -->



## 3) **DSPy's Ecosystem** advances open-source AI research.

Compared to monolithic LMs, DSPy's modular paradigm enables a large community to improve the compositional architectures, inference-time strategies, and optimizers for LM programs in an open, distributed way. This gives DSPy users more control, helps them iterate much faster, and allows their programs to get better over time by applying the latest optimizers or modules.

The DSPy research effort started at Stanford NLP in Feb 2022, building on what we had learned from developing early [compound LM systems](https://bair.berkeley.edu/blog/2024/02/18/compound-ai-systems/) like [ColBERT-QA](https://arxiv.org/abs/2007.00814), [Baleen](https://arxiv.org/abs/2101.00436), and [Hindsight](https://arxiv.org/abs/2110.07752). The first version was released as [DSP](https://arxiv.org/abs/2212.14024) in Dec 2022 and evolved by Oct 2023 into [DSPy](https://arxiv.org/abs/2310.03714). Thanks to [250 contributors](https://github.com/stanfordnlp/dspy/graphs/contributors), DSPy has introduced tens of thousands of people to building and optimizing modular LM programs.

Since then, DSPy's community has produced a large body of work on optimizers, like [MIPROv2](https://arxiv.org/abs/2406.11695), [BetterTogether](https://arxiv.org/abs/2407.10930), and [LeReT](https://arxiv.org/abs/2410.23214), on program architectures, like [STORM](https://arxiv.org/abs/2402.14207), [IReRa](https://arxiv.org/abs/2401.12178), and [DSPy Assertions](https://arxiv.org/abs/2312.13382), and on successful applications to new problems, like [PAPILLON](https://arxiv.org/abs/2410.17127), [PATH](https://arxiv.org/abs/2406.11706), [WangLab@MEDIQA](https://arxiv.org/abs/2404.14544), [UMD's Prompting Case Study](https://arxiv.org/abs/2406.06608), and [Haize's Red-Teaming Program](https://blog.haizelabs.com/posts/dspy/), in addition to many open-source projects, production applications, and other [use cases](/dspy-usecases/).


## End of index.md

---


## File: dspy-usecases.md

# DSPy Use Cases

We often get questions like "How are people using DSPy in practice?", both in production and for research. This list was created to collect a few pointers and to encourage others in the community to add their own work below.

This list is ever expanding and highly incomplete (WIP)! We'll be adding a bunch more. If you would like to add your product or research to this list, please make a PR.

## A Few Company Use Cases

| **Name** | **Use Cases** |
|---|---|
| **[JetBlue](https://www.jetblue.com/)** | Multiple chatbot use cases. [Blog](https://www.databricks.com/blog/optimizing-databricks-llm-pipelines-dspy) |
| **[Replit](https://replit.com/)** | Synthesize diffs using code LLMs using a DSPy pipeline. [Blog](https://blog.replit.com/code-repair) |
| **[Databricks](https://www.databricks.com/)** | Research, products, and customer solutions around LM Judges, RAG, classification, and other applications. [Blog](https://www.databricks.com/blog/dspy-databricks) [Blog II](https://www.databricks.com/customers/ddi) |
| **[Sephora](https://www.sephora.com/)** | Undisclosed agent usecases; perspectives shared in [DAIS Session](https://www.youtube.com/watch?v=D2HurSldDkE). |
| **[Zoro UK](https://www.zoro.co.uk/)** | E-commerce applications around structured shopping. [Portkey Session](https://www.youtube.com/watch?v=_vGKSc1tekE) |
| **[VMware](https://www.vmware.com/)** | RAG and other prompt optimization applications. [Interview in The Register.](https://www.theregister.com/2024/02/22/prompt_engineering_ai_models/) [Business Insider.](https://www.businessinsider.com/chaptgpt-large-language-model-ai-prompt-engineering-automated-optimizer-2024-3) |
| **[Haize Labs](https://www.haizelabs.com/)** | Automated red-teaming for LLMs. [Blog](https://blog.haizelabs.com/posts/dspy/) |
| **[Plastic Labs](https://www.plasticlabs.ai/)** | Different pipelines within Honcho. [Blog](https://blog.plasticlabs.ai/blog/User-State-is-State-of-the-Art) |
| **[PingCAP](https://pingcap.com/)** | Building a knowledge graph. [Article](https://www.pingcap.com/article/building-a-graphrag-from-wikipedia-page-using-dspy-openai-and-tidb-vector-database/) |
| **[Salomatic](https://langtrace.ai/blog/case-study-how-salomatic-used-langtrace-to-build-a-reliable-medical-report-generation-system)** | Enriching medical reports using DSPy. [Blog](https://langtrace.ai/blog/case-study-how-salomatic-used-langtrace-to-build-a-reliable-medical-report-generation-system) |
| **[Truelaw](https://www.youtube.com/watch?v=O0F3RAWZNfM)** | How Truelaw builds bespoke LLM pipelines for law firms using DSPy. [Podcast](https://www.youtube.com/watch?v=O0F3RAWZNfM) |
| **[Moody's](https://www.moodys.com/)** | Leveraging DSPy to optimize RAG systems, LLM-as-a-Judge, and agentic systems for financial workflows. |
| **[Normal Computing](https://www.normalcomputing.com/)** | Translate specs from chip companies from English to intermediate formal languages |
| **[Procure.FYI](https://www.procure.fyi/)** | Process messy, publicly available technology spending and pricing data via DSPy. |
| **[RadiantLogic](https://www.radiantlogic.com/)** | AI Data Assistant. DSPy is used for the agent that routes the query, the context extraction module, the text-to-sql conversion engine, and the table summarization module. |
| **[Raia](https://raiahealth.com/)** | Using DSPy for AI-powered Personal Healthcare Agents. |
| **[Hyperlint](https://hyperlint.com)** | Uses DSPy to generate technical documentation. DSPy helps to fetch relevant information and synthesize that into tutorials. |
| **[Starops](https://staropshq.com/) & [Saya](https://heysaya.ai/)** | Building research documents given a user's corpus. Generate prompts to create more articles from example articles. |
| **[Tessel AI](https://tesselai.com/)** | Enhancing human-machine interaction with data use cases. |
| **[Dicer.ai](https://dicer.ai/)** | Uses DSPy for marketing AI to get the most from their paid ads. |
| **[Howie](https://howie.ai)** | Using DSPy to automate meeting scheduling through email. |
| **[Isoform.ai](https://isoform.ai)** | Building custom integrations using DSPy. |
| **[Trampoline AI](https://trampoline.ai)** | Uses DSPy to power their data-augmentation and LM pipelines. |
| **[Pretrain](https://pretrain.com)** | Uses DSPy to automatically optimize AI performance towards user-defined tasks based on uploaded examples. |

WIP. This list mainly includes companies that have public posts or have OKed being included for specific products so far.


## A Few Papers Using DSPy

| **Name** | **Description** |
|---|---|
| **[STORM](https://arxiv.org/abs/2402.14207)** | Writing Wikipedia-like Articles From Scratch. |
| **[PATH](https://arxiv.org/abs/2406.11706)** | Prompts as Auto-Optimized Training Hyperparameters: Training Best-in-Class IR Models from Scratch with 10 Gold Labels |
| **[WangLab @ MEDIQA](https://arxiv.org/abs/2404.14544)** | UofT's winning system at MEDIQA, outperforms the next best system by 20 points |
| **[UMD's Suicide Detection System](https://arxiv.org/abs/2406.06608)** | Outperforms 20-hour expert human prompt engineering by 40% |
| **[IReRa](https://arxiv.org/abs/2401.12178)** | Infer-Retrieve-Rank: Extreme Classification with > 10,000 Labels |
| **[Unreasonably Effective Eccentric Prompts](https://arxiv.org/abs/2402.10949v2)** | General Prompt Optimization |
| **[Palimpzest](https://arxiv.org/abs/2405.14696)** | A Declarative System for Optimizing AI Workloads |
| **[AI Agents that Matter](https://arxiv.org/abs/2407.01502v1)** | Agent Efficiency Optimization |
| **[EDEN](https://arxiv.org/abs/2406.17982v1)** | Empathetic Dialogues for English Learning: Uses adaptive empathetic feedback to improve student grit |
| **[ECG-Chat](https://arxiv.org/pdf/2408.08849)** | Uses DSPy with GraphRAG for medical report generation |
| **[DSPy Assertions](https://arxiv.org/abs/2312.13382)** | Various applications of imposing hard and soft constraints on LM outputs |
| **[DSPy Guardrails](https://boxiyu.github.io/assets/pdf/DSPy_Guardrails.pdf)** | Reduce the attack success rate of CodeAttack, decreasing from 75% to 5% |
| **[Co-STORM](https://arxiv.org/pdf/2408.15232)** | Collaborative STORM: Generate Wikipedia-like articles through collaborative discourse among users and multiple LM agents |

## A Few Repositories (or other OSS examples) using DSPy

| **Name** | **Description/Link** |
|---|---|
| **Stanford CS 224U Homework** | [GitHub](https://github.com/cgpotts/cs224u/blob/main/hw_openqa.ipynb) |
| **STORM Report Generation (10,000 GitHub stars)** | [GitHub](https://github.com/stanford-oval/storm) |
| **DSPy Redteaming** | [GitHub](https://github.com/haizelabs/dspy-redteam) |
| **DSPy Theory of Mind** |  [GitHub](https://github.com/plastic-labs/dspy-opentom) |
| **Indic cross-lingual Natural Language Inference** |  [GitHub](https://github.com/saifulhaq95/DSPy-Indic/blob/main/indicxlni.ipynb) |
| **Optimizing LM for Text2SQL using DSPy** | [GitHub](https://github.com/jjovalle99/DSPy-Text2SQL) |
| **DSPy PII Masking Demo by Eric Ness** | [Colab](https://colab.research.google.com/drive/1KZR1sGTp_RLWUJPAiK1FKPKI-Qn9neUm?usp=sharing) |
| **DSPy on BIG-Bench Hard Example** |  [GitHub](https://drchrislevy.github.io/posts/dspy/dspy.html) |
| **Building a chess playing agent using DSPy** |  [GitHub](https://medium.com/thoughts-on-machine-learning/building-a-chess-playing-agent-using-dspy-9b87c868f71e) |
| **Ittia Research Fact Checking** | [GitHub](https://github.com/ittia-research/check) |
| **Strategic Debate via Tree-of-Thought** | [GitHub](https://github.com/zbambergerNLP/strategic-debate-tot) |
| **Sanskrit to English Translation App**| [GitHub](https://github.com/ganarajpr/sanskrit-translator-dspy) |
| **DSPy for extracting features from PDFs on arXiv**| [GitHub](https://github.com/S1M0N38/dspy-arxiv) |
| **DSPygen: DSPy in Ruby on Rails**| [GitHub](https://github.com/seanchatmangpt/dspygen) |
| **DSPy Inspector**| [GitHub](https://github.com/Neoxelox/dspy-inspector) |
| **DSPy with FastAPI**| [GitHub](https://github.com/diicellman/dspy-rag-fastapi) |
| **DSPy for Indian Languages**| [GitHub](https://github.com/saifulhaq95/DSPy-Indic) |
| **Hurricane: Blog Posts with Generative Feedback Loops!**| [GitHub](https://github.com/weaviate-tutorials/Hurricane) |
| **RAG example using DSPy, Gradio, FastAPI, and Ollama**| [GitHub](https://github.com/diicellman/dspy-gradio-rag) |
| **Synthetic Data Generation**| [GitHub](https://colab.research.google.com/drive/1CweVOu0qhTC0yOfW5QkLDRIKuAuWJKEr?usp=sharing) |
| **Self Discover**| [GitHub](https://colab.research.google.com/drive/1GkAQKmw1XQgg5UNzzy8OncRe79V6pADB?usp=sharing) |

TODO: This list in particular is highly incomplete. There are a couple dozen other good ones.

## A Few Providers, Integrations, and related Blog Releases

| **Name** | **Link** |
|---|---|
| **Databricks** | [Link](https://www.databricks.com/blog/dspy-databricks) |
| **Zenbase** | [Link](https://zenbase.ai/) |
| **LangWatch** | [Link](https://langwatch.ai/blog/introducing-dspy-visualizer) |
| **Gradient** | [Link](https://gradient.ai/blog/achieving-gpt-4-level-performance-at-lower-cost-using-dspy) |
| **Snowflake** | [Link](https://medium.com/snowflake/dspy-snowflake-140d6d947d73) |
| **Langchain** | [Link](https://python.langchain.com/v0.2/docs/integrations/providers/dspy/) |
| **Weaviate** | [Link](https://weaviate.io/blog/dspy-optimizers) |
| **Qdrant** | [Link](https://qdrant.tech/documentation/frameworks/dspy/) |
| **Weights & Biases Weave** | [Link](https://weave-docs.wandb.ai/guides/integrations/dspy/) |
| **Milvus** | [Link](https://milvus.io/docs/integrate_with_dspy.md) |
| **Neo4j** | [Link](https://neo4j.com/labs/genai-ecosystem/dspy/) |
| **Lightning AI** | [Link](https://lightning.ai/lightning-ai/studios/dspy-programming-with-foundation-models) |
| **Haystack** | [Link](https://towardsdatascience.com/automating-prompt-engineering-with-dspy-and-haystack-926a637a3f43) |
| **Arize** | [Link](https://docs.arize.com/phoenix/tracing/integrations-tracing/dspy) |
| **LlamaIndex** | [Link](https://github.com/stanfordnlp/dspy/blob/main/examples/llamaindex/dspy_llamaindex_rag.ipynb) |
| **Langtrace** | [Link](https://docs.langtrace.ai/supported-integrations/llm-frameworks/dspy) |
| **Langfuse** | [Link](https://langfuse.com/docs/integrations/dspy) |
| **OpenLIT** | [Link](https://docs.openlit.io/latest/integrations/dspy) |

## A Few Blogs & Videos on using DSPy

| **Name** | **Link** |
|---|---|
| **Blog Posts** | |
| **Why I bet on DSPy** | [Blog](https://blog.isaacbmiller.com/posts/dspy) |
| **Not Your Average Prompt Engineering** | [Blog](https://jina.ai/news/dspy-not-your-average-prompt-engineering/) |
| **Why I'm excited about DSPy** | [Blog](https://substack.stephen.so/p/why-im-excited-about-dspy) |
| **Achieving GPT-4 Performance at Lower Cost** | [Link](https://gradient.ai/blog/achieving-gpt-4-level-performance-at-lower-cost-using-dspy) |
| **Prompt engineering is a task best left to AI models** | [Link](https://www.theregister.com/2024/02/22/prompt_engineering_ai_models/) |
| **What makes DSPy a valuable framework for developing complex language model pipelines?** | [Link](https://medium.com/@sujathamudadla1213/what-makes-dspy-a-valuable-framework-for-developing-complex-language-model-pipelines-edfa5b4bcf9b) |
| **DSPy: A new framework to program your foundation models just by prompting** | [Link](https://www.linkedin.com/pulse/dspy-new-framework-program-your-foundation-models-just-prompting-lli4c/) |
| **Intro to DSPy: Goodbye Prompting, Hello Programming** | [Link](https://medium.com/towards-data-science/intro-to-dspy-goodbye-prompting-hello-programming-4ca1c6ce3eb9) |
| **DSPyGen: Revolutionizing AI** | [Link](https://www.linkedin.com/pulse/launch-alert-dspygen-20242252-revolutionizing-ai-sean-chatman--g9f1c/) |
| **Building an AI Assistant with DSPy** | [Link](https://www.linkedin.com/pulse/building-ai-assistant-dspy-valliappa-lakshmanan-vgnsc/) |
| **Videos** | |
| **DSPy Explained! (60K views)** | [Link](https://www.youtube.com/watch?v=41EfOY0Ldkc) |
| **DSPy Intro from Sephora (25K views)** | [Link](https://www.youtube.com/watch?v=D2HurSldDkE) |
| **Structured Outputs with DSPy** | [Link](https://www.youtube.com/watch?v=tVw3CwrN5-8) |
| **DSPy and ColBERT - Weaviate Podcast** | [Link](https://www.youtube.com/watch?v=CDung1LnLbY) |
| **SBTB23 DSPy** | [Link](https://www.youtube.com/watch?v=Dt3H2ninoeY) |
| **Optimization with DSPy and LangChain** | [Link](https://www.youtube.com/watch?v=4EXOmWeqXRc) |
| **Automated Prompt Engineering + Visualization** | [Link](https://www.youtube.com/watch?v=eAZ2LtJ6D5k) |
| **Transforming LM Calls into Pipelines** | [Link](https://www.youtube.com/watch?v=NoaDWKHdkHg) |
| **NeurIPS Hacker Cup: DSPy for Code Gen** | [Link](https://www.youtube.com/watch?v=gpe-rtJN8z8) |
| **MIPRO and DSPy - Weaviate Podcast** | [Link](https://www.youtube.com/watch?v=skMH3DOV_UQ) |
| **Getting Started with RAG in DSPy** | [Link](https://www.youtube.com/watch?v=CEuUG4Umfxs) |
| **Adding Depth to DSPy Programs** | [Link](https://www.youtube.com/watch?v=0c7Ksd6BG88) |
| **Programming Foundation Models with DSPy** | [Link](https://www.youtube.com/watch?v=Y94tw4eDHW0) |
| **DSPy End-to-End: SF Meetup** | [Link](https://www.youtube.com/watch?v=Y81DoFmt-2U) |
| **Monitoring & Tracing DSPy with Langtrace** | [Link](https://langtrace.ai/blog/announcing-dspy-support-in-langtrace) |
| **Teaching chat models to solve chess puzzles using DSPy + Finetuning** | [Link](https://raw.sh/posts/chess_puzzles) |

TODO: This list in particular is highly incomplete. There are dozens of other good ones. To allow space, divide into opintionated blogs / podcasts / interviews vs. tutorials & talks.

Credit: Some of these resources were originally compiled in the [Awesome DSPy](https://github.com/ganarajpr/awesome-dspy/tree/master) repo.

### Weaviate has a directory of 10 amazing notebooks and 6 podcasts!

Huge shoutout to them for the massive support ❤️. See the [Weaviate DSPy directory](https://weaviate.io/developers/weaviate/more-resources/dspy).


## End of dspy-usecases.md

---


## File: learn/8-typed_predictors.md

# Typed Predictors

In DSPy Signatures, we have `InputField` and `OutputField` that define the nature of inputs and outputs of the field. However, the inputs and output to these fields are always `str`-typed, which requires input and output string processing.

Pydantic `BaseModel` is a great way to enforce type constraints on the fields, but it is not directly compatible with the `dspy.Signature`. Typed Predictors resolves this as a way to enforce the type constraints on the inputs and outputs of the fields in a `dspy.Signature`.

## Executing Typed Predictors

Using Typed Predictors is not too different than any other module with the minor additions of type hints to signature attributes and using a special Predictor module instead of `dspy.Predict`. Let's take a look at a simple example to understand this.

### Defining Input and Output Models

Let's take a simple task as an example i.e. given the `context` and `query`, the LLM should return an `answer` and `confidence_score`. Let's define our `Input` and `Output` models via pydantic.

```python
from pydantic import BaseModel, Field

class Input(BaseModel):
    context: str = Field(description="The context for the question")
    query: str = Field(description="The question to be answered")

class Output(BaseModel):
    answer: str = Field(description="The answer for the question")
    confidence: float = Field(ge=0, le=1, description="The confidence score for the answer")
```

As you can see, we can describe the attributes by defining a simple Signature that takes in the input and returns the output.

### Creating Typed Predictor

A Typed Predictor needs a Typed Signature, which extends a `dspy.Signature` with the addition of specifying "field type".

```python
class QASignature(dspy.Signature):
    """Answer the question based on the context and query provided, and on the scale of 10 tell how confident you are about the answer."""

    input: Input = dspy.InputField()
    output: Output = dspy.OutputField()
```

Now that we have the `QASignature`, let's define a Typed Predictor that executes this Signature while conforming to the type constraints.

```python
predictor = dspy.TypedPredictor(QASignature)
```

Similar to other modules, we pass the `QASignature` to `dspy.TypedPredictor` which enforces the typed constraints.

And similarly to `dspy.Predict`, we can also use a "string signature", which we type as:
```python
predictor = dspy.TypedPredictor("input:Input -> output:Output")
```

### I/O in Typed Predictors

Now let's test out the Typed Predictor by providing some sample input to the predictor and verifying the output type. We can create an `Input` instance and pass it to the predictor to get a dictionary of the output. 

```python
doc_query_pair = Input(
    context="The quick brown fox jumps over the lazy dog",
    query="What does the fox jumps over?",
)

prediction = predictor(input=doc_query_pair)
```

Let's see the output and its type.

```python
answer = prediction.output.answer
confidence_score = prediction.output.confidence

print(f"Prediction: {prediction}\n\n")
print(f"Answer: {answer}, Answer Type: {type(answer)}")
print(f"Confidence Score: {confidence_score}, Confidence Score Type: {type(confidence_score)}")
```

## Typed Chain of Thoughts with `dspy.TypedChainOfThought`

Extending the analogous comparison of `TypedPredictor` to `dspy.Predict`, we create `TypedChainOfThought`, the typed counterpart of `dspy.ChainOfThought`:

```python
cot_predictor = dspy.TypedChainOfThought(QASignature)

doc_query_pair = Input(
    context="The quick brown fox jumps over the lazy dog",
    query="What does the fox jumps over?",
)

prediction = cot_predictor(input=doc_query_pair)
```

## Typed Predictors as Decorators

While the `dspy.TypedPredictor` and `dspy.TypedChainOfThought` provide a convenient way to use typed predictors, you can also use them as decorators to enforce type constraints on the inputs and outputs of the function. This relies on the internal definitions of the Signature class and its function arguments, outputs, and docstrings.

```python
@dspy.predictor
def answer(doc_query_pair: Input) -> Output:
    """Answer the question based on the context and query provided, and on the scale of 0-1 tell how confident you are about the answer."""
    pass

@dspy.cot
def answer(doc_query_pair: Input) -> Output:
    """Answer the question based on the context and query provided, and on the scale of 0-1 tell how confident you are about the answer."""
    pass

prediction = answer(doc_query_pair=doc_query_pair)
```

## Composing Functional Typed Predictors in `dspy.Module`

If you're creating DSPy pipelines via `dspy.Module`, then you can simply use Functional Typed Predictors by creating these class methods and using them as decorators. Here is an example of using functional typed predictors to create a `SimplifiedBaleen` pipeline:

```python
class SimplifiedBaleen(FunctionalModule):
    def __init__(self, passages_per_hop=3, max_hops=1):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=passages_per_hop)
        self.max_hops = max_hops

    @cot
    def generate_query(self, context: list[str], question) -> str:
        """Write a simple search query that will help answer a complex question."""
        pass

    @cot
    def generate_answer(self, context: list[str], question) -> str:
        """Answer questions with short factoid answers."""
        pass

    def forward(self, question):
        context = []

        for _ in range(self.max_hops):
            query = self.generate_query(context=context, question=question)
            passages = self.retrieve(query).passages
            context = deduplicate(context + passages)

        answer = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=answer)
```

## Optimizing Typed Predictors

Typed predictors can be optimized on the Signature instructions through the `optimize_signature` optimizer. Here is an example of this optimization on the `QASignature`:

```python
import dspy
from dspy.evaluate import Evaluate
from dspy.evaluate.metrics import answer_exact_match
from dspy.teleprompt.signature_opt_typed import optimize_signature

turbo = dspy.OpenAI(model='gpt-3.5-turbo', max_tokens=4000)
gpt4 = dspy.OpenAI(model='gpt-4', max_tokens=4000)
dspy.settings.configure(lm=turbo)

evaluator = Evaluate(devset=devset, metric=answer_exact_match, num_threads=10, display_progress=True)

result = optimize_signature(
    student=dspy.TypedPredictor(QASignature),
    evaluator=evaluator,
    initial_prompts=6,
    n_iterations=100,
    max_examples=30,
    verbose=True,
    prompt_model=gpt4,
)
```


## End of learn/8-typed_predictors.md

---


## File: learn/index.md

---
sidebar_position: 1
---

# Learning DSPy: An Overview

DSPy exposes a very small API that you can learn quickly. However, building a new AI system is a more open-ended journey of iterative development, in which you compose the tools and design patterns of DSPy to optimize for _your_ objectives. The three stages of building AI systems in DSPy are:

1) **DSPy Programming.** This is about defining your task, its constraints, exploring a few examples, and using that to inform your initial pipeline design.

2) **DSPy Evaluation.** Once your system starts working, this is the stage where you collect an initial development set, define your DSPy metric, and use these to iterate on your system more systematically.

3) **DSPy Optimization.** Once you have a way to evaluate your system, you use DSPy optimizers to tune the prompts or weights in your program.

We suggest learning and applying DSPy in this order. For example, it's unproductive to launch optimization runs using a poorly-design program or a bad metric.

## End of learn/index.md

---


## File: learn/optimization/overview.md

---
sidebar_position: 1
---


# Optimization in DSPy

Once you have a system and a way to evaluate it, you can use DSPy optimizers to tune the prompts or weights in your program. Now it's useful to expand your data collection effort into building a training set and a held-out test set, in addition to the development set you've been using for exploration. For the training set (and its subset, validation set), you can often get substantial value out of 30 examples, but aim for at least 300 examples. Some optimizers accept a `trainset` only. Others ask for a `trainset` and a `valset`. For prompt optimizers, we suggest starting with a 20% split for training and 80% for validation, which is often the _opposite_ of what one does for DNNs.

After your first few optimization runs, you are either very happy with everything or you've made a lot of progress but you don't like something about the final program or the metric. At this point, go back to step 1 (Programming in DSPy) and revisit the major questions. Did you define your task well? Do you need to collect (or find online) more data for your problem? Do you want to update your metric? And do you want to use a more sophisticated optimizer? Do you need to consider advanced features like DSPy Assertions? Or, perhaps most importantly, do you want to add some more complexity or steps in your DSPy program itself? Do you want to use multiple optimizers in a sequence?

Iterative development is key. DSPy gives you the pieces to do that incrementally: iterating on your data, your program structure, your assertions, your metric, and your optimization steps. Optimizing complex LM programs is an entirely new paradigm that only exists in DSPy at the time of writing (update: there are now numerous DSPy extension frameworks, so this part is no longer true :-), so naturally the norms around what to do are still emerging. If you need help, we recently created a [Discord server](https://discord.gg/XCGy2WDCQB) for the community.



## End of learn/optimization/overview.md

---


## File: learn/optimization/optimizers.md

---
sidebar_position: 1
---

# DSPy Optimizers (formerly Teleprompters)


A **DSPy optimizer** is an algorithm that can tune the parameters of a DSPy program (i.e., the prompts and/or the LM weights) to maximize the metrics you specify, like accuracy.


A typical DSPy optimizer takes three things:

- Your **DSPy program**. This may be a single module (e.g., `dspy.Predict`) or a complex multi-module program.

- Your **metric**. This is a function that evaluates the output of your program, and assigns it a score (higher is better).

- A few **training inputs**. This may be very small (i.e., only 5 or 10 examples) and incomplete (only inputs to your program, without any labels).

If you happen to have a lot of data, DSPy can leverage that. But you can start small and get strong results.

**Note:** Formerly called teleprompters. We are making an official name update, which will be reflected throughout the library and documentation.


## What does a DSPy Optimizer tune? How does it tune them?

Different optimizers in DSPy will tune your program's quality by **synthesizing good few-shot examples** for every module, like `dspy.BootstrapRS`,<sup>[1](https://arxiv.org/abs/2310.03714)</sup> **proposing and intelligently exploring better natural-language instructions** for every prompt, like `dspy.MIPROv2`,<sup>[2](https://arxiv.org/abs/2406.11695)</sup> and **building datasets for your modules and using them to finetune the LM weights** in your system, like `dspy.BootstrapFinetune`.<sup>[3](https://arxiv.org/abs/2407.10930)</sup>

??? "What's an example of a DSPy optimizer? How do different optimizers work?"

    Take the `dspy.MIPROv2` optimizer as an example. First, MIPRO starts with the **bootstrapping stage**. It takes your program, which may be unoptimized at this point, and runs it many times across different inputs to collect traces of input/output behavior for each one of your modules. It filters these traces to keep only those that appear in trajectories scored highly by your metric. Second, MIPRO enters its **grounded proposal stage**. It previews your DSPy program's code, your data, and traces from running your program, and uses them to draft many potential instructions for every prompt in your program. Third, MIPRO launches the **discrete search stage**. It samples mini-batches from your training set, proposes a combination of instructions and traces to use for constructing every prompt in the pipeline, and evaluates the candidate program on the mini-batch. Using the resulting score, MIPRO updates a surrogate model that helps the proposals get better over time.

    One thing that makes DSPy optimizers so powerful is that they can be composed. You can run `dspy.MIPROv2` and use the produced program as an input to `dspy.MIPROv2` again or, say, to `dspy.BootstrapFinetune` to get better results. This is partly the essence of `dspy.BetterTogether`. Alternatively, you can run the optimizer and then extract the top-5 candidate programs and build a `dspy.Ensemble` of them. This allows you to scale _inference-time compute_ (e.g., ensembles) as well as DSPy's unique _pre-inference time compute_ (i.e., optimization budget) in highly systematic ways.



## What DSPy Optimizers are currently available?

Optimizers can be accessed via `from dspy.teleprompt import *`.

### Automatic Few-Shot Learning

These optimizers extend the signature by automatically generating and including **optimized** examples within the prompt sent to the model, implementing few-shot learning.

1. **`LabeledFewShot`**: Simply constructs few-shot examples (demos) from provided labeled input and output data points.  Requires `k` (number of examples for the prompt) and `trainset` to randomly select `k` examples from.

2. **`BootstrapFewShot`**: Uses a `teacher` module (which defaults to your program) to generate complete demonstrations for every stage of your program, along with labeled examples in `trainset`. Parameters include `max_labeled_demos` (the number of demonstrations randomly selected from the `trainset`) and `max_bootstrapped_demos` (the number of additional examples generated by the `teacher`). The bootstrapping process employs the metric to validate demonstrations, including only those that pass the metric in the "compiled" prompt. Advanced: Supports using a `teacher` program that is a *different* DSPy program that has compatible structure, for harder tasks.

3. [**`BootstrapFewShotWithRandomSearch`**](/deep-dive/optimizers/bootstrap-fewshot): Applies `BootstrapFewShot` several times with random search over generated demonstrations, and selects the best program over the optimization. Parameters mirror those of `BootstrapFewShot`, with the addition of `num_candidate_programs`, which specifies the number of random programs evaluated over the optimization, including candidates of the uncompiled program, `LabeledFewShot` optimized program, `BootstrapFewShot` compiled program with unshuffled examples and `num_candidate_programs` of `BootstrapFewShot` compiled programs with randomized example sets.

4. **`KNNFewShot`**. Uses k-Nearest Neighbors algorithm to find the nearest training example demonstrations for a given input example. These nearest neighbor demonstrations are then used as the trainset for the BootstrapFewShot optimization process. See [this notebook](https://github.com/stanfordnlp/dspy/blob/main/examples/knn.ipynb) for an example.


### Automatic Instruction Optimization

These optimizers produce optimal instructions for the prompt and, in the case of MIPROv2 can also optimize the set of few-shot demonstrations.

5. [**`COPRO`**](/deep-dive/optimizers/copro): Generates and refines new instructions for each step, and optimizes them with coordinate ascent (hill-climbing using the metric function and the `trainset`). Parameters include `depth` which is the number of iterations of prompt improvement the optimizer runs over.

6. [**`MIPROv2`**](/deep-dive/optimizers/miprov2): Generates instructions *and* few-shot examples in each step. The instruction generation is data-aware and demonstration-aware. Uses Bayesian Optimization to effectively search over the space of generation instructions/demonstrations across your modules.


### Automatic Finetuning

This optimizer is used to fine-tune the underlying LLM(s).

7. **`BootstrapFinetune`**: Distills a prompt-based DSPy program into weight updates. The output is a DSPy program that has the same steps, but where each step is conducted by a finetuned model instead of a prompted LM.


### Program Transformations

8. **`Ensemble`**: Ensembles a set of DSPy programs and either uses the full set or randomly samples a subset into a single program.


## Which optimizer should I use?

Ultimately, finding the ‘right’ optimizer to use & the best configuration for your task will require experimentation. Success in DSPy is still an iterative process - getting the best performance on your task will require you to explore and iterate.  

That being said, here's the general guidance on getting started:

- If you have **very few examples** (around 10), start with `BootstrapFewShot`.
- If you have **more data** (50 examples or more), try  `BootstrapFewShotWithRandomSearch`.
- If you prefer to do **instruction optimization only** (i.e. you want to keep your prompt 0-shot), use `MIPROv2` [configured for 0-shot optimization to optimize](/deep-dive/optimizers/miprov2#optimizing-instructions-only-with-miprov2-0-shot). 
- If you’re willing to use more inference calls to perform **longer optimization runs** (e.g. 40 trials or more), and have enough data (e.g. 200 examples or more to prevent overfitting) then try `MIPROv2`. 
- If you have been able to use one of these with a large LM (e.g., 7B parameters or above) and need a very **efficient program**, finetune a small LM for your task with `BootstrapFinetune`.

## How do I use an optimizer?

They all share this general interface, with some differences in the keyword arguments (hyperparameters). Detailed documentation for key optimizers can be found [here](/deep-dive/optimizers/vfrs), and a full list can be found [here](/cheatsheet).

Let's see this with the most common one, `BootstrapFewShotWithRandomSearch`.

```python
from dspy.teleprompt import BootstrapFewShotWithRandomSearch

# Set up the optimizer: we want to "bootstrap" (i.e., self-generate) 8-shot examples of your program's steps.
# The optimizer will repeat this 10 times (plus some initial attempts) before selecting its best attempt on the devset.
config = dict(max_bootstrapped_demos=4, max_labeled_demos=4, num_candidate_programs=10, num_threads=4)

teleprompter = BootstrapFewShotWithRandomSearch(metric=YOUR_METRIC_HERE, **config)
optimized_program = teleprompter.compile(YOUR_PROGRAM_HERE, trainset=YOUR_TRAINSET_HERE)
```


!!! info "Getting Started III: Optimizing the LM prompts or weights in DSPy programs"
    A typical simple optimization run costs on the order of $2 USD and takes around ten minutes, but be careful when running optimizers with very large LMs or very large datasets.
    Optimizer runs can cost as little as a few cents or up to tens of dollars, depending on your LM, dataset, and configuration.
    
    === "Optimizing prompts for a ReAct agent"
        This is a minimal but fully runnable example of setting up a `dspy.ReAct` agent that answers questions via
        search from Wikipedia and then optimizing it using `dspy.MIPROv2` in the cheap `light` mode on 500
        question-answer pairs sampled from the `HotPotQA` dataset.

        ```python linenums="1"
        import dspy
        from dspy.datasets import HotPotQA

        dspy.configure(lm=dspy.LM('openai/gpt-4o-mini'))

        def search(query: str) -> list[str]:
            """Retrieves abstracts from Wikipedia."""
            results = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')(query, k=3)
            return [x['text'] for x in results]

        trainset = [x.with_inputs('question') for x in HotPotQA(train_seed=2024, train_size=500).train]
        react = dspy.ReAct("question -> answer", tools=[search])

        tp = dspy.MIPROv2(metric=dspy.evaluate.answer_exact_match, auto="light", num_threads=24)
        optimized_react = tp.compile(react, trainset=trainset)
        ```

        An informal run similar to this on DSPy 2.5.29 raises ReAct's score from 24% to 51%.

    === "Optimizing prompts for RAG"
        Given a retrieval index to `search`, your favorite `dspy.LM`, and a small `trainset` of questions and ground-truth responses, the following code snippet can optimize your RAG system with long outputs against the built-in `dspy.SemanticF1` metric, which is implemented as a DSPy module.

        ```python linenums="1"
        class RAG(dspy.Module):
            def __init__(self, num_docs=5):
                self.num_docs = num_docs
                self.respond = dspy.ChainOfThought('context, question -> response')

            def forward(self, question):
                context = search(question, k=self.num_docs)   # not defined in this snippet, see link above
                return self.respond(context=context, question=question)

        tp = dspy.MIPROv2(metric=dspy.SemanticF1(), auto="medium", num_threads=24)
        optimized_rag = tp.compile(RAG(), trainset=trainset, max_bootstrapped_demos=2, max_labeled_demos=2)
        ```

        For a complete RAG example that you can run, start this [tutorial](http://127.0.0.1:8000/quick-start/getting-started-01/). It improves the quality of a RAG system over a subset of StackExchange communities from 53% to 61%.

    === "Optimizing weights for Classification"
        This is a minimal but fully runnable example of setting up a `dspy.ChainOfThought` module that classifies
        short texts into one of 77 banking labels and then using `dspy.BootstrapFinetune` with 2000 text-label pairs
        from the `Banking77` to finetune the weights of GPT-4o-mini for this task. We use the variant
        `dspy.ChainOfThoughtWithHint`, which takes an optional `hint` at bootstrapping time, to maximize the utility of
        the training data. Naturally, hints are not available at test time.

        <details><summary>Click to show dataset setup code.</summary>

        ```python linenums="1"
        import random
        from typing import Literal
        from dspy.datasets import DataLoader
        from datasets import load_dataset

        # Load the Banking77 dataset.
        CLASSES = load_dataset("PolyAI/banking77", split="train", trust_remote_code=True).features['label'].names
        kwargs = dict(fields=("text", "label"), input_keys=("text",), split="train", trust_remote_code=True)

        # Load the first 2000 examples from the dataset, and assign a hint to each *training* example.
        trainset = [
            dspy.Example(x, hint=CLASSES[x.label], label=CLASSES[x.label]).with_inputs("text", "hint")
            for x in DataLoader().from_huggingface(dataset_name="PolyAI/banking77", **kwargs)[:2000]
        ]
        random.Random(0).shuffle(trainset)
        ```
        </details>

        ```python linenums="1"
        import dspy
        dspy.configure(lm=dspy.LM('gpt-4o-mini-2024-07-18'))
        
        # Define the DSPy module for classification. It will use the hint at training time, if available.
        signature = dspy.Signature("text -> label").with_updated_fields('label', type_=Literal[tuple(CLASSES)])
        classify = dspy.ChainOfThoughtWithHint(signature)

        # Optimize via BootstrapFinetune.
        optimizer = dspy.BootstrapFinetune(metric=(lambda x, y, trace=None: x.label == y.label), num_threads=24)
        optimized = optimizer.compile(classify, trainset=trainset)

        optimized(text="What does a pending cash withdrawal mean?")
        ```

        **Possible Output (from the last line):**
        ```text
        Prediction(
            reasoning='A pending cash withdrawal indicates that a request to withdraw cash has been initiated but has not yet been completed or processed. This status means that the transaction is still in progress and the funds have not yet been deducted from the account or made available to the user.',
            label='pending_cash_withdrawal'
        )
        ```

        An informal run similar to this on DSPy 2.5.29 raises GPT-4o-mini's score 66% to 87%.


## Saving and loading optimizer output

After running a program through an optimizer, it's useful to also save it. At a later point, a program can be loaded from a file and used for inference. For this, the `load` and `save` methods can be used.

```python
optimized_program.save(YOUR_SAVE_PATH)
```

The resulting file is in plain-text JSON format. It contains all the parameters and steps in the source program. You can always read it and see what the optimizer generated. You can add `save_field_meta` to additionally save the list of fields with the keys, `name`, `field_type`, `description`, and `prefix` with: `optimized_program.save(YOUR_SAVE_PATH, save_field_meta=True).


To load a program from a file, you can instantiate an object from that class and then call the load method on it.

```python
loaded_program = YOUR_PROGRAM_CLASS()
loaded_program.load(path=YOUR_SAVE_PATH)
```



## End of learn/optimization/optimizers.md

---


## File: learn/optimization/solving_your_task.md

---
sidebar_position: 1
---

# Using DSPy in 8 Steps

Using DSPy well for solving a new task is just doing good machine learning with LMs. It's an iterative process: you make some initial choices, which will be sub-optimal, and then you refine them incrementally. 

As we discuss below, you will define your task and the metrics you want to maximize, and prepare a few example inputs — typically without labels (or only with labels for the final outputs, if your metric requires them). Then, you build your pipeline by selecting built-in layers [(`modules`)](/building-blocks/3-modules) to use, giving each layer a [`signature` (input/output spec)](/building-blocks/2-signatures), and then calling your modules freely in your Python code. Lastly, you use a DSPy [`optimizer`](/building-blocks/6-optimizers) to compile your code into high-quality instructions, automatic few-shot examples, or updated LM weights for your LM.


## 1) Define your task.

You cannot use DSPy well if you haven't defined the problem you're trying to solve.

**Expected Input/Output Behavior:** Are you trying to build a chatbot over your data? A code assistant? A system for extracting information from papers? Or perhaps a translation system? Or a system for highlighting snippets from search results? Or a system to summarize information on a topic, with citations?

It's often useful to come up with just 3-4 examples of the inputs and outputs of your program (e.g., questions and their answers, or topics and their summaries). If you need help thinking about your task, we recently created a [Discord server](https://discord.gg/XCGy2WDCQB) for the community.

**Quality and Cost Specs:** You probably don't have infinite budget. Your final system can't be too expensive to run, and it should probably respond to users quickly enough.

Take this as an opportunity to guess what kind of language model you'd like to use. Maybe GPT-3.5? Or a small open model, like Mistral-7B or Llama2-13B-chat? Or Mixtral? Or maybe you really need GPT-4-turbo? Or perhaps your resources are very constrained, and you want your final LM to be T5-base.

## 2) Define your pipeline.

What should your DSPy program do? Can it just be a simple chain-of-thought step? Or do you need the LM to use retrieval? Or maybe other tools, like a calculator or a calendar API?

Is there a typical workflow for solving your problem in multiple well-defined steps? Or do you want a fully open-ended LM (or open-ended tool use with agents) for your task?

Think about this space but always start simple. Almost every task should probably start with just a single [dspy.ChainofThought](/deep-dive/modules/chain-of-thought) module, and then add complexity incrementally as you go.

Then write your (initial) DSPy program. Again: start simple, and let the next few steps guide any complexity you will add.

## 3) Explore a few examples.

By this point, you probably have a few examples of the task you're trying to solve.

Run them through your pipeline. Consider using a large and powerful LM at this point, or a couple of different LMs, just to understand what's possible. (DSPy will make swapping these LMs pretty easy - [LM Guide](/building-blocks/1-language_models).)

At this point, you're still using your pipeline zero-shot, so it will be far from perfect. DSPy will help you optimize the instructions, few-shot examples, and even weights of your LM calls below, but understanding where things go wrong in zero-shot usage will go a long way.

Record the interesting (both easy and hard) examples you try: even if you don't have labels, simply tracking the inputs you tried will be useful for DSPy optimizers below.

## 4) Define your data.

Now it's time to more formally declare your training and validation data for DSPy evaluation and optimization - [Data Guide](/building-blocks/4-data).

You can use DSPy optimizers usefully with as few as 10 examples, but having 50-100 examples (or even better, 300-500 examples) goes a long way.

How can you get examples like these? If your task is extremely unusual, please invest in preparing ~10 examples by hand. Often times, depending on your metric below, you just need inputs and not labels, so it's not that hard.

However, chances are that your task is not actually that unique. You can almost always find somewhat adjacent datasets on, say, HuggingFace datasets or other forms of data that you can leverage here.

If there's data whose licenses are permissive enough, we suggest you use them. Otherwise, you can also start using/deploying/demoing your system and collect some initial data that way.


## 5) Define your metric.

What makes outputs from your system good or bad? Invest in defining metrics and improving them over time incrementally. It's really hard to consistently improve what you aren't able to define.

A metric is just a function that will take examples from your data and take the output of your system, and return a score that quantifies how good the output is - [Metric Guide](/building-blocks/5-metrics).

For simple tasks, this could be just "accuracy" or "exact match" or "F1 score". This may be the case for simple classification or short-form QA tasks.

However, for most applications, your system will output long-form outputs. There, your metric should probably be a smaller DSPy program that checks multiple properties of the output (quite possibly using AI feedback from LMs).

Getting this right on the first try is unlikely, but you should start with something simple and iterate. (If your metric is itself a DSPy program, notice that one of the most powerful ways to iterate is to compile (optimize) your metric itself. That's usually easy because the output of the metric is usually a simple value (e.g., a score out of 5) so the metric's metric is easy to define and optimize by collecting a few examples.)


## 6) Collect preliminary "zero-shot" evaluations.

Now that you have some data and a metric, run evaluation on your pipeline before any optimizer runs.

Look at the outputs and the metric scores. This will probably allow you to spot any major issues, and it will define a baseline for your next step.

## 7) Compile with a DSPy optimizer.

Given some data and a metric, we can now optimize the program you built - [Optimizer Guide](/building-blocks/6-optimizers).

DSPy includes many optimizers that do different things. Remember: DSPy optimizers will create examples of each step, craft instructions, and/or update LM weights. In general, you don't need to have labels for your pipeline steps, but your data examples need to have input values and whatever labels your metric requires (e.g., no labels if your metric is reference-free, but final output labels otherwise in most cases).

Here's the general guidance on getting started:

* If you have very little data, e.g. 10 examples of your task, use [`BootstrapFewShot`](/deep-dive/optimizers/bootstrap-fewshot)

* If you have slightly more data, e.g. 50 examples of your task, use [`BootstrapFewShotWithRandomSearch`](/deep-dive/optimizers/bfrs).

* If you have more data than that, e.g. 300 examples or more, use [`MIPRO`](/deep-dive/optimizers/miprov2).

* If you have been able to use one of these with a large LM (e.g., 7B parameters or above) and need a very efficient program, compile that down to a small LM with `BootstrapFinetune`.

## 8) Iterate.

At this point, you are either very happy with everything (we've seen quite a few people get it right on first try with DSPy) or, more likely, you've made a lot of progress but you don't like something about the final program or the metric.

At this point, go back to step 1 and revisit the major questions. Did you define your task well? Do you need to collect (or find online) more data for your problem? Do you want to update your metric? And do you want to use a more sophisticated optimizer? Do you need to consider advanced features like [DSPy Assertions](/building-blocks/7-assertions)? Or, perhaps most importantly, do you want to add some more complexity or steps in your DSPy program itself? Do you want to use multiple optimizers in a sequence?

Iterative development is key. DSPy gives you the pieces to do that incrementally: iterating on your data, your program structure, your assertions, your metric, and your optimization steps.

Optimizing complex LM programs is an entirely new paradigm that only exists in DSPy at the time of writing, so naturally the norms around what to do are still emerging. If you need help, we recently created a [Discord server](https://discord.gg/XCGy2WDCQB) for the community.


## End of learn/optimization/solving_your_task.md

---


## File: learn/programming/signatures.md

---
sidebar_position: 2
---

# Signatures

When we assign tasks to LMs in DSPy, we specify the behavior we need as a Signature.

**A signature is a declarative specification of input/output behavior of a DSPy module.** Signatures allow you to tell the LM _what_ it needs to do, rather than specify _how_ we should ask the LM to do it.

You're probably familiar with function signatures, which specify the input and output arguments and their types. DSPy signatures are similar, but with a couple of differences. While typical function signatures just _describe_ things, DSPy Signatures _declare and initialize the behavior_ of modules. Moreover, the field names matter in DSPy Signatures. You express semantic roles in plain English: a `question` is different from an `answer`, a `sql_query` is different from `python_code`.

## Why should I use a DSPy Signature?

For modular and clean code, in which LM calls can be optimized into high-quality prompts (or automatic finetunes). Most people coerce LMs to do tasks by hacking long, brittle prompts. Or by collecting/generating data for fine-tuning. Writing signatures is far more modular, adaptive, and reproducible than hacking at prompts or finetunes. The DSPy compiler will figure out how to build a highly-optimized prompt for your LM (or finetune your small LM) for your signature, on your data, and within your pipeline. In many cases, we found that compiling leads to better prompts than humans write. Not because DSPy optimizers are more creative than humans, but simply because they can try more things and tune the metrics directly.

## **Inline** DSPy Signatures

Signatures can be defined as a short string, with argument names and optional types that define semantic roles for inputs/outputs.

1. Question Answering: `"question -> answer"`, which is equivalent to `"question: str -> answer: str"` as the default type is always `str`

2. Sentiment Classification: `"sentence -> sentiment: bool"`, e.g. `True` if positive

3. Summarization: `"document -> summary"`

Your signatures can also have multiple input/output fields with types:

4. Retrieval-Augmented Question Answering: `"context: list[str], question: str -> answer: str"`

5. Multiple-Choice Question Answering with Reasoning: `"question, choices: list[str] -> reasoning: str, selection: int"`

**Tip:** For fields, any valid variable names work! Field names should be semantically meaningful, but start simple and don't prematurely optimize keywords! Leave that kind of hacking to the DSPy compiler. For example, for summarization, it's probably fine to say `"document -> summary"`, `"text -> gist"`, or `"long_context -> tldr"`.


### Example A: Sentiment Classification

```python
sentence = "it's a charming and often affecting journey."  # example from the SST-2 dataset.

classify = dspy.Predict('sentence -> sentiment: bool')  # we'll see an example with Literal[] later
classify(sentence=sentence).sentiment
```
**Output:**
```text
True
```

### Example B: Summarization

```python
# Example from the XSum dataset.
document = """The 21-year-old made seven appearances for the Hammers and netted his only goal for them in a Europa League qualification round match against Andorran side FC Lustrains last season. Lee had two loan spells in League One last term, with Blackpool and then Colchester United. He scored twice for the U's but was unable to save them from relegation. The length of Lee's contract with the promoted Tykes has not been revealed. Find all the latest football transfers on our dedicated page."""

summarize = dspy.ChainOfThought('document -> summary')
response = summarize(document=document)

print(response.summary)
```
**Possible Output:**
```text
The 21-year-old Lee made seven appearances and scored one goal for West Ham last season. He had loan spells in League One with Blackpool and Colchester United, scoring twice for the latter. He has now signed a contract with Barnsley, but the length of the contract has not been revealed.
```

Many DSPy modules (except `dspy.Predict`) return auxiliary information by expanding your signature under the hood.

For example, `dspy.ChainOfThought` also adds a `reasoning` field that includes the LM's reasoning before it generates the output `summary`.

```python
print("Reasoning:", response.reasoning)
```
**Possible Output:**
```text
Reasoning: We need to highlight Lee's performance for West Ham, his loan spells in League One, and his new contract with Barnsley. We also need to mention that his contract length has not been disclosed.
```

## **Class-based** DSPy Signatures

For some advanced tasks, you need more verbose signatures. This is typically to:

1. Clarify something about the nature of the task (expressed below as a `docstring`).

2. Supply hints on the nature of an input field, expressed as a `desc` keyword argument for `dspy.InputField`.

3. Supply constraints on an output field, expressed as a `desc` keyword argument for `dspy.OutputField`.

### Example C: Classification

```python
from typing import Literal

class Emotion(dspy.Signature):
    """Classify emotion."""
    
    sentence: str = dspy.InputField()
    sentiment: Literal['sadness', 'joy', 'love', 'anger', 'fear', 'surprise'] = dspy.OutputField()

sentence = "i started feeling a little vulnerable when the giant spotlight started blinding me"  # from dair-ai/emotion

classify = dspy.Predict(Emotion)
classify(sentence=sentence)
```
**Possible Output:**
```text
Prediction(
    sentiment='fear'
)
```

**Tip:** There's nothing wrong with specifying your requests to the LM more clearly. Class-based Signatures help you with that. However, don't prematurely tune the keywords of your signature by hand. The DSPy optimizers will likely do a better job (and will transfer better across LMs).

### Example D: A metric that evaluates faithfulness to citations

```python
class CheckCitationFaithfulness(dspy.Signature):
    """Verify that the text is based on the provided context."""

    context: str = dspy.InputField(desc="facts here are assumed to be true")
    text: str = dspy.InputField()
    faithfulness: bool = dspy.OutputField()
    evidence: dict[str, list[str]] = dspy.OutputField(desc="Supporting evidence for claims")

context = "The 21-year-old made seven appearances for the Hammers and netted his only goal for them in a Europa League qualification round match against Andorran side FC Lustrains last season. Lee had two loan spells in League One last term, with Blackpool and then Colchester United. He scored twice for the U's but was unable to save them from relegation. The length of Lee's contract with the promoted Tykes has not been revealed. Find all the latest football transfers on our dedicated page."

text = "Lee scored 3 goals for Colchester United."

faithfulness = dspy.ChainOfThought(CheckCitationFaithfulness)
faithfulness(context=context, text=text)
```
**Possible Output:**
```text
Prediction(
    reasoning="Let's check the claims against the context. The text states Lee scored 3 goals for Colchester United, but the context clearly states 'He scored twice for the U's'. This is a direct contradiction.",
    faithfulness=False,
    evidence={'goal_count': ["scored twice for the U's"]}
)
```

### Example E: Multi-modal image classification

```python
class DogPictureSignature(dspy.Signature):
    """Output the dog breed of the dog in the image."""
    image_1: dspy.Image = dspy.InputField(desc="An image of a dog")
    answer: str = dspy.OutputField(desc="The dog breed of the dog in the image")

image_url = "https://picsum.photos/id/237/200/300"
classify = dspy.Predict(DogPictureSignature)
classify(image_1=dspy.Image.from_url(image_url))
```

**Possible Output:**

```text
Prediction(
    answer='Labrador Retriever'
)
```

## Using signatures to build modules & compiling them

While signatures are convenient for prototyping with structured inputs/outputs, that's not the only reason to use them!

You should compose multiple signatures into bigger [DSPy modules](modules.md) and [compile these modules into optimized prompts](../optimization/optimizers.md) and finetunes.


## End of learn/programming/signatures.md

---


## File: learn/programming/overview.md

---
sidebar_position: 1
---

# Programming in DSPy

DSPy is a bet on _writing code instead of strings_. In other words, building the right control flow is crucial. Start by **defining your task**. What are the inputs to your system and what should your system produce as output? Is it a chatbot over your data or perhaps a code assistant? Or maybe a system for translation, for highlighting snippets from search results, or for generating reports with citations?

Next, **define your initial pipeline**. Can your DSPy program just be a single module or do you need to break it down into a few steps? Do you need retrieval or other tools, like a calculator or a calendar API? Is there a typical workflow for solving your problem in multiple well-scoped steps, or do you want more open-ended tool use with agents for your task? Think about these but start simple, perhaps with just a single `dspy.ChainOfThought` module, then add complexity incrementally based on observations.

As you do this, **craft and try a handful of examples** of the inputs to your program. Consider using a powerful LM at this point, or a couple of different LMs, just to understand what's possible. Record interesting (both easy and hard) examples you try. This will be useful when you are doing evaluation and optimization later.


??? "Beyond encouraging good design patterns, how does DSPy help here?"

    Conventional prompts couple your fundamental system architecture with incidental choices not portable to new LMs, objectives, or pipelines. A conventional prompt asks the LM to take some inputs and produce some outputs of certain types (a _signature_), formats the inputs in certain ways and requests outputs in a form it can parse accurately (an _adapter_), asks the LM to apply certain strategies like "thinking step by step" or using tools (a _module_'s logic), and relies on substantial trial-and-error to discover the right way to ask each LM to do this (a form of manual _optimization_).
    
    DSPy separates these concerns and automates the lower-level ones until you need to consider them. This allow you to write much shorter code, with much higher portability. For example, if you write a program using DSPy modules, you can swap the LM or its adapter without changing the rest of your logic. Or you can exchange one _module_, like `dspy.ChainOfThought`, with another, like `dspy.ProgramOfThought`, without modifying your signatures. When you're ready to use optimizers, the same program can have its prompts optimized or its LM weights fine-tuned.


## End of learn/programming/overview.md

---


## File: learn/programming/modules.md

---
sidebar_position: 3
---

# Modules

A **DSPy module** is a building block for programs that use LMs.

- Each built-in module abstracts a **prompting technique** (like chain of thought or ReAct). Crucially, they are generalized to handle any signature.

- A DSPy module has **learnable parameters** (i.e., the little pieces comprising the prompt and the LM weights) and can be invoked (called) to process inputs and return outputs.

- Multiple modules can be composed into bigger modules (programs). DSPy modules are inspired directly by NN modules in PyTorch, but applied to LM programs.


## How do I use a built-in module, like `dspy.Predict` or `dspy.ChainOfThought`?

Let's start with the most fundamental module, `dspy.Predict`. Internally, all other DSPy modules are built using `dspy.Predict`. We'll assume you are already at least a little familiar with [DSPy signatures](/building-blocks/signatures), which are declarative specs for defining the behavior of any module we use in DSPy.

To use a module, we first **declare** it by giving it a signature. Then we **call** the module with the input arguments, and extract the output fields!

```python
sentence = "it's a charming and often affecting journey."  # example from the SST-2 dataset.

# 1) Declare with a signature.
classify = dspy.Predict('sentence -> sentiment: bool')

# 2) Call with input argument(s). 
response = classify(sentence=sentence)

# 3) Access the output.
print(response.sentiment)
```
**Output:**
```text
True
```

When we declare a module, we can pass configuration keys to it.

Below, we'll pass `n=5` to request five completions. We can also pass `temperature` or `max_len`, etc.

Let's use `dspy.ChainOfThought`. In many cases, simply swapping `dspy.ChainOfThought` in place of `dspy.Predict` improves quality.

```python
question = "What's something great about the ColBERT retrieval model?"

# 1) Declare with a signature, and pass some config.
classify = dspy.ChainOfThought('question -> answer', n=5)

# 2) Call with input argument.
response = classify(question=question)

# 3) Access the outputs.
response.completions.answer
```
**Possible Output:**
```text
['One great thing about the ColBERT retrieval model is its superior efficiency and effectiveness compared to other models.',
 'Its ability to efficiently retrieve relevant information from large document collections.',
 'One great thing about the ColBERT retrieval model is its superior performance compared to other models and its efficient use of pre-trained language models.',
 'One great thing about the ColBERT retrieval model is its superior efficiency and accuracy compared to other models.',
 'One great thing about the ColBERT retrieval model is its ability to incorporate user feedback and support complex queries.']
```

Let's discuss the output object here. The `dspy.ChainOfThought` module will generally inject a `reasoning` before the output field(s) of your signature.

Let's inspect the (first) reasoning and answer!

```python
print(f"Reasoning: {response.reasoning}")
print(f"Answer: {response.answer}")
```
**Possible Output:**
```text
Reasoning: We can consider the fact that ColBERT has shown to outperform other state-of-the-art retrieval models in terms of efficiency and effectiveness. It uses contextualized embeddings and performs document retrieval in a way that is both accurate and scalable.
Answer: One great thing about the ColBERT retrieval model is its superior efficiency and effectiveness compared to other models.
```

This is accessible whether we request one or many completions.

We can also access the different completions as a list of `Prediction`s or as several lists, one for each field.

```python
response.completions[3].reasoning == response.completions.reasoning[3]
```
**Output:**
```text
True
```


## What other DSPy modules are there? How can I use them?

The others are very similar. They mainly change the internal behavior with which your signature is implemented!

1. **`dspy.Predict`**: Basic predictor. Does not modify the signature. Handles the key forms of learning (i.e., storing the instructions and demonstrations and updates to the LM).

2. **`dspy.ChainOfThought`**: Teaches the LM to think step-by-step before committing to the signature's response.

3. **`dspy.ProgramOfThought`**: Teaches the LM to output code, whose execution results will dictate the response.

4. **`dspy.ReAct`**: An agent that can use tools to implement the given signature.

5. **`dspy.MultiChainComparison`**: Can compare multiple outputs from `ChainOfThought` to produce a final prediction.


We also have some function-style modules:

6. **`dspy.majority`**: Can do basic voting to return the most popular response from a set of predictions.


!!! info "A few examples of DSPy modules on simple tasks."
    Try the examples below after configuring your `lm`. Adjust the fields to explore what tasks your LM can do well out of the box.

    === "Math"

        ```python linenums="1"
        math = dspy.ChainOfThought("question -> answer: float")
        math(question="Two dice are tossed. What is the probability that the sum equals two?")
        ```
        
        **Possible Output:**
        ```text
        Prediction(
            reasoning='When two dice are tossed, each die has 6 faces, resulting in a total of 6 x 6 = 36 possible outcomes. The sum of the numbers on the two dice equals two only when both dice show a 1. This is just one specific outcome: (1, 1). Therefore, there is only 1 favorable outcome. The probability of the sum being two is the number of favorable outcomes divided by the total number of possible outcomes, which is 1/36.',
            answer=0.0277776
        )
        ```

    === "Retrieval-Augmented Generation"

        ```python linenums="1"       
        def search(query: str) -> list[str]:
            """Retrieves abstracts from Wikipedia."""
            results = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')(query, k=3)
            return [x['text'] for x in results]
        
        rag = dspy.ChainOfThought('context, question -> response')

        question = "What's the name of the castle that David Gregory inherited?"
        rag(context=search(question), question=question)
        ```
        
        **Possible Output:**
        ```text
        Prediction(
            reasoning='The context provides information about David Gregory, a Scottish physician and inventor. It specifically mentions that he inherited Kinnairdy Castle in 1664. This detail directly answers the question about the name of the castle that David Gregory inherited.',
            response='Kinnairdy Castle'
        )
        ```

    === "Classification"

        ```python linenums="1"
        from typing import Literal

        class Classify(dspy.Signature):
            """Classify sentiment of a given sentence."""
            
            sentence: str = dspy.InputField()
            sentiment: Literal['positive', 'negative', 'neutral'] = dspy.OutputField()
            confidence: float = dspy.OutputField()

        classify = dspy.Predict(Classify)
        classify(sentence="This book was super fun to read, though not the last chapter.")
        ```
        
        **Possible Output:**

        ```text
        Prediction(
            sentiment='positive',
            confidence=0.75
        )
        ```

    === "Information Extraction"

        ```python linenums="1"        
        text = "Apple Inc. announced its latest iPhone 14 today. The CEO, Tim Cook, highlighted its new features in a press release."

        module = dspy.Predict("text -> title, headings: list[str], entities_and_metadata: list[dict[str, str]]")
        response = module(text=text)

        print(response.title)
        print(response.headings)
        print(response.entities_and_metadata)
        ```
        
        **Possible Output:**
        ```text
        Apple Unveils iPhone 14
        ['Introduction', 'Key Features', "CEO's Statement"]
        [{'entity': 'Apple Inc.', 'type': 'Organization'}, {'entity': 'iPhone 14', 'type': 'Product'}, {'entity': 'Tim Cook', 'type': 'Person'}]
        ```

    === "Agents"

        ```python linenums="1"       
        def evaluate_math(expression: str) -> float:
            return dspy.PythonInterpreter({}).execute(expression)

        def search_wikipedia(query: str) -> str:
            results = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')(query, k=3)
            return [x['text'] for x in results]

        react = dspy.ReAct("question -> answer: float", tools=[evaluate_math, search_wikipedia])

        pred = react(question="What is 9362158 divided by the year of birth of David Gregory of Kinnairdy castle?")
        print(pred.answer)
        ```
        
        **Possible Output:**

        ```text
        5761.328
        ```


## How do I compose multiple modules into a bigger program?

DSPy is just Python code that uses modules in any control flow you like, with a little magic internally at `compile` time to trace your LM calls. What this means is that, you can just call the modules freely.

See tutorials like [multi-hop search](https://dspy.ai/tutorials/multihop_search/), whose module is reproduced below as an example.

```python linenums="1"        
class Hop(dspy.Module):
    def __init__(self, num_docs=10, num_hops=4):
        self.num_docs, self.num_hops = num_docs, num_hops
        self.generate_query = dspy.ChainOfThought('claim, notes -> query')
        self.append_notes = dspy.ChainOfThought('claim, notes, context -> new_notes: list[str], titles: list[str]')

    def forward(self, claim: str) -> list[str]:
        notes = []
        titles = []

        for _ in range(self.num_hops):
            query = self.generate_query(claim=claim, notes=notes).query
            context = search(query, k=self.num_docs)
            prediction = self.append_notes(claim=claim, notes=notes, context=context)
            notes.extend(prediction.new_notes)
            titles.extend(prediction.titles)
        
        return dspy.Prediction(notes=notes, titles=list(set(titles)))
```


## End of learn/programming/modules.md

---


## File: learn/programming/7-assertions.md

# DSPy Assertions 

!!! warning "This page is outdated and may not be fully accurate in DSPy 2.5"


## Introduction

Language models (LMs) have transformed how we interact with machine learning, offering vast capabilities in natural language understanding and generation. However, ensuring these models adhere to domain-specific constraints remains a challenge. Despite the growth of techniques like fine-tuning or “prompt engineering”, these approaches are extremely tedious and rely on heavy, manual hand-waving to guide the LMs in adhering to specific constraints. Even DSPy's modularity of programming prompting pipelines lacks mechanisms to effectively and automatically enforce these constraints. 

To address this, we introduce DSPy Assertions, a feature within the DSPy framework designed to automate the enforcement of computational constraints on LMs. DSPy Assertions empower developers to guide LMs towards desired outcomes with minimal manual intervention, enhancing the reliability, predictability, and correctness of LM outputs.

### dspy.Assert and dspy.Suggest API    

We introduce two primary constructs within DSPy Assertions:

- **`dspy.Assert`**:
  - **Parameters**: 
    - `constraint (bool)`: Outcome of Python-defined boolean validation check.
    - `msg (Optional[str])`: User-defined error message providing feedback or correction guidance.
    - `backtrack (Optional[module])`: Specifies target module for retry attempts upon constraint failure. The default backtracking module is the last module before the assertion.
  - **Behavior**: Initiates retry  upon failure, dynamically adjusting the pipeline's execution. If failures persist, it halts execution and raises a `dspy.AssertionError`.

- **`dspy.Suggest`**:
  - **Parameters**: Similar to `dspy.Assert`.
  - **Behavior**: Encourages self-refinement through retries without enforcing hard stops. Logs failures after maximum backtracking attempts and continues execution.

- **dspy.Assert vs. Python Assertions**: Unlike conventional Python `assert` statements that terminate the program upon failure, `dspy.Assert` conducts a sophisticated retry mechanism, allowing the pipeline to adjust. 

Specifically, when a constraint is not met:

- Backtracking Mechanism: An under-the-hood backtracking is initiated, offering the model a chance to self-refine and proceed, which is done through signature modification.
- Dynamic Signature Modification: internally modifying your DSPy program’s Signature by adding the following fields:
    - Past Output: your model's past output that did not pass the validation_fn
    - Instruction: your user-defined feedback message on what went wrong and what possibly to fix

If the error continues past the `max_backtracking_attempts`, then `dspy.Assert` will halt the pipeline execution, alerting you with an `dspy.AssertionError`. This ensures your program doesn't continue executing with “bad” LM behavior and immediately highlights sample failure outputs for user assessment.

- **dspy.Suggest vs. dspy.Assert**: `dspy.Suggest` on the other hand offers a softer approach. It maintains the same retry backtracking as `dspy.Assert` but instead serves as a gentle nudger. If the model outputs cannot pass the model constraints after the `max_backtracking_attempts`, `dspy.Suggest` will log the persistent failure and continue execution of the program on the rest of the data. This ensures the LM pipeline works in a "best-effort" manner without halting execution. 

- **`dspy.Suggest`** statements are best utilized as "helpers" during the evaluation phase, offering guidance and potential corrections without halting the pipeline.
- **`dspy.Assert`** statements are recommended during the development stage as "checkers" to ensure the LM behaves as expected, providing a robust mechanism for identifying and addressing errors early in the development cycle.


## Use Case: Including Assertions in DSPy Programs

We start with using an example of a multi-hop QA SimplifiedBaleen pipeline as defined in the intro walkthrough. 

```python
class SimplifiedBaleen(dspy.Module):
    def __init__(self, passages_per_hop=2, max_hops=2):
        super().__init__()

        self.generate_query = [dspy.ChainOfThought(GenerateSearchQuery) for _ in range(max_hops)]
        self.retrieve = dspy.Retrieve(k=passages_per_hop)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
        self.max_hops = max_hops

    def forward(self, question):
        context = []
        prev_queries = [question]

        for hop in range(self.max_hops):
            query = self.generate_query[hop](context=context, question=question).query
            prev_queries.append(query)
            passages = self.retrieve(query).passages
            context = deduplicate(context + passages)
        
        pred = self.generate_answer(context=context, question=question)
        pred = dspy.Prediction(context=context, answer=pred.answer)
        return pred

baleen = SimplifiedBaleen()

baleen(question = "Which award did Gary Zukav's first book receive?")
```

To include DSPy Assertions, we simply define our validation functions and declare our assertions following the respective model generation. 

For this use case, suppose we want to impose the following constraints:
    1. Length - each query should be less than 100 characters
    2. Uniqueness - each generated query should differ from previously-generated queries. 
    
We can define these validation checks as boolean functions:

```python
#simplistic boolean check for query length
len(query) <= 100

#Python function for validating distinct queries
def validate_query_distinction_local(previous_queries, query):
    """check if query is distinct from previous queries"""
    if previous_queries == []:
        return True
    if dspy.evaluate.answer_exact_match_str(query, previous_queries, frac=0.8):
        return False
    return True
```

We can declare these validation checks through `dspy.Suggest` statements (as we want to test the program in a best-effort demonstration). We want to keep these after the query generation `query = self.generate_query[hop](context=context, question=question).query`.

```python
dspy.Suggest(
    len(query) <= 100,
    "Query should be short and less than 100 characters",
    target_module=self.generate_query
)

dspy.Suggest(
    validate_query_distinction_local(prev_queries, query),
    "Query should be distinct from: "
    + "; ".join(f"{i+1}) {q}" for i, q in enumerate(prev_queries)),
    target_module=self.generate_query
)
```

It is recommended to define a program with assertions separately than your original program if you are doing comparative evaluation for the effect of assertions. If not, feel free to set Assertions away!

Let's take a look at how the SimplifiedBaleen program will look with Assertions included:

```python
class SimplifiedBaleenAssertions(dspy.Module):
    def __init__(self, passages_per_hop=2, max_hops=2):
        super().__init__()
        self.generate_query = [dspy.ChainOfThought(GenerateSearchQuery) for _ in range(max_hops)]
        self.retrieve = dspy.Retrieve(k=passages_per_hop)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
        self.max_hops = max_hops

    def forward(self, question):
        context = []
        prev_queries = [question]

        for hop in range(self.max_hops):
            query = self.generate_query[hop](context=context, question=question).query

            dspy.Suggest(
                len(query) <= 100,
                "Query should be short and less than 100 characters",
                target_module=self.generate_query
            )

            dspy.Suggest(
                validate_query_distinction_local(prev_queries, query),
                "Query should be distinct from: "
                + "; ".join(f"{i+1}) {q}" for i, q in enumerate(prev_queries)),
                target_module=self.generate_query
            )

            prev_queries.append(query)
            passages = self.retrieve(query).passages
            context = deduplicate(context + passages)
        
        if all_queries_distinct(prev_queries):
            self.passed_suggestions += 1

        pred = self.generate_answer(context=context, question=question)
        pred = dspy.Prediction(context=context, answer=pred.answer)
        return pred
```

Now calling programs with DSPy Assertions requires one last step, and that is transforming the program to wrap it with internal assertions backtracking and Retry logic. 

```python
from dspy.primitives.assertions import assert_transform_module, backtrack_handler

baleen_with_assertions = assert_transform_module(SimplifiedBaleenAssertions(), backtrack_handler)

# backtrack_handler is parameterized over a few settings for the backtracking mechanism
# To change the number of max retry attempts, you can do
baleen_with_assertions_retry_once = assert_transform_module(SimplifiedBaleenAssertions(), 
    functools.partial(backtrack_handler, max_backtracks=1))
```

Alternatively, you can also directly call `activate_assertions` on the program with `dspy.Assert/Suggest` statements using the default backtracking mechanism (`max_backtracks=2`):

```python
baleen_with_assertions = SimplifiedBaleenAssertions().activate_assertions()
```

Now let's take a look at the internal LM backtracking by inspecting the history of the LM query generations. Here we see that when a query fails to pass the validation check of being less than 100 characters, its internal `GenerateSearchQuery` signature is dynamically modified during the backtracking+Retry process to include the past query and the corresponding user-defined instruction: `"Query should be short and less than 100 characters"`.


```text
Write a simple search query that will help answer a complex question.

---

Follow the following format.

Context: may contain relevant facts

Question: ${question}

Reasoning: Let's think step by step in order to ${produce the query}. We ...

Query: ${query}

---

Context:
[1] «Kerry Condon | Kerry Condon (born 4 January 1983) is [...]»
[2] «Corona Riccardo | Corona Riccardo (c. 1878October 15, 1917) was [...]»

Question: Who acted in the shot film The Shore and is also the youngest actress ever to play Ophelia in a Royal Shakespeare Company production of "Hamlet." ?

Reasoning: Let's think step by step in order to find the answer to this question. First, we need to identify the actress who played Ophelia in a Royal Shakespeare Company production of "Hamlet." Then, we need to find out if this actress also acted in the short film "The Shore."

Query: "actress who played Ophelia in Royal Shakespeare Company production of Hamlet" + "actress in short film The Shore"



Write a simple search query that will help answer a complex question.

---

Follow the following format.

Context: may contain relevant facts

Question: ${question}

Past Query: past output with errors

Instructions: Some instructions you must satisfy

Query: ${query}

---

Context:
[1] «Kerry Condon | Kerry Condon (born 4 January 1983) is an Irish television and film actress, best known for her role as Octavia of the Julii in the HBO/BBC series "Rome," as Stacey Ehrmantraut in AMC's "Better Call Saul" and as the voice of F.R.I.D.A.Y. in various films in the Marvel Cinematic Universe. She is also the youngest actress ever to play Ophelia in a Royal Shakespeare Company production of "Hamlet."»
[2] «Corona Riccardo | Corona Riccardo (c. 1878October 15, 1917) was an Italian born American actress who had a brief Broadway stage career before leaving to become a wife and mother. Born in Naples she came to acting in 1894 playing a Mexican girl in a play at the Empire Theatre. Wilson Barrett engaged her for a role in his play "The Sign of the Cross" which he took on tour of the United States. Riccardo played the role of Ancaria and later played Berenice in the same play. Robert B. Mantell in 1898 who struck by her beauty also cast her in two Shakespeare plays, "Romeo and Juliet" and "Othello". Author Lewis Strang writing in 1899 said Riccardo was the most promising actress in America at the time. Towards the end of 1898 Mantell chose her for another Shakespeare part, Ophelia im Hamlet. Afterwards she was due to join Augustin Daly's Theatre Company but Daly died in 1899. In 1899 she gained her biggest fame by playing Iras in the first stage production of Ben-Hur.»

Question: Who acted in the shot film The Shore and is also the youngest actress ever to play Ophelia in a Royal Shakespeare Company production of "Hamlet." ?

Past Query: "actress who played Ophelia in Royal Shakespeare Company production of Hamlet" + "actress in short film The Shore"

Instructions: Query should be short and less than 100 characters

Query: "actress Ophelia RSC Hamlet" + "actress The Shore"

```


## Assertion-Driven Optimizations

DSPy Assertions work with optimizations that DSPy offers, particularly with `BootstrapFewShotWithRandomSearch`, including the following settings:

- Compilation with Assertions
    This includes assertion-driven example bootstrapping and counterexample bootstrapping during compilation. The teacher model for bootstrapping few-shot demonstrations can make use of DSPy Assertions to offer robust bootstrapped examples for the student model to learn from during inference. In this setting, the student model does not perform assertion aware optimizations (backtracking and retry) during inference.
- Compilation + Inference with Assertions
    -This includes assertion-driven optimizations in both compilation and inference. Now the teacher model offers assertion-driven examples but the student can further optimize with assertions of its own during inference time. 
```python
teleprompter = BootstrapFewShotWithRandomSearch(
    metric=validate_context_and_answer_and_hops,
    max_bootstrapped_demos=max_bootstrapped_demos,
    num_candidate_programs=6,
)

#Compilation with Assertions
compiled_with_assertions_baleen = teleprompter.compile(student = baleen, teacher = baleen_with_assertions, trainset = trainset, valset = devset)

#Compilation + Inference with Assertions
compiled_baleen_with_assertions = teleprompter.compile(student=baleen_with_assertions, teacher = baleen_with_assertions, trainset=trainset, valset=devset)

```

## End of learn/programming/7-assertions.md

---


## File: learn/programming/language_models.md

---
sidebar_position: 2
---

# Language Models

The first step in any DSPy code is to set up your language model. For example, you can configure OpenAI's GPT-4o-mini as your default LM as follows.

```python linenums="1"
# Authenticate via `OPENAI_API_KEY` env: import os; os.environ['OPENAI_API_KEY'] = 'here'
lm = dspy.LM('openai/gpt-4o-mini')
dspy.configure(lm=lm)
```

!!! info "A few different LMs"

    === "OpenAI"
        You can authenticate by setting the `OPENAI_API_KEY` env variable or passing `api_key` below.

        ```python linenums="1"
        import dspy
        lm = dspy.LM('openai/gpt-4o-mini', api_key='YOUR_OPENAI_API_KEY')
        dspy.configure(lm=lm)
        ```

    === "Anthropic"
        You can authenticate by setting the ANTHROPIC_API_KEY env variable or passing `api_key` below.

        ```python linenums="1"
        import dspy
        lm = dspy.LM('anthropic/claude-3-opus-20240229', api_key='YOUR_ANTHROPIC_API_KEY')
        dspy.configure(lm=lm)
        ```

    === "Databricks"
        If you're on the Databricks platform, authentication is automatic via their SDK. If not, you can set the env variables `DATABRICKS_API_KEY` and `DATABRICKS_API_BASE`, or pass `api_key` and `api_base` below.

        ```python linenums="1"
        import dspy
        lm = dspy.LM('databricks/databricks-meta-llama-3-1-70b-instruct')
        dspy.configure(lm=lm)
        ```

    === "Local LMs on a GPU server"
          First, install [SGLang](https://sgl-project.github.io/start/install.html) and launch its server with your LM.

          ```bash
          > pip install "sglang[all]"
          > pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/ 

          > CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server --port 7501 --model-path meta-llama/Meta-Llama-3-8B-Instruct
          ```

          Then, connect to it from your DSPy code as an OpenAI-compatible endpoint.

          ```python linenums="1"
          lm = dspy.LM("openai/meta-llama/Meta-Llama-3-8B-Instruct",
                           api_base="http://localhost:7501/v1",  # ensure this points to your port
                           api_key="", model_type='chat')
          dspy.configure(lm=lm)
          ```

    === "Local LMs on your laptop"
          First, install [Ollama](https://github.com/ollama/ollama) and launch its server with your LM.

          ```bash
          > curl -fsSL https://ollama.ai/install.sh | sh
          > ollama run llama3.2:1b
          ```

          Then, connect to it from your DSPy code.

        ```python linenums="1"
        import dspy
        lm = dspy.LM('ollama_chat/llama3.2', api_base='http://localhost:11434', api_key='')
        dspy.configure(lm=lm)
        ```

    === "Other providers"
        In DSPy, you can use any of the dozens of [LLM providers supported by LiteLLM](https://docs.litellm.ai/docs/providers). Simply follow their instructions for which `{PROVIDER}_API_KEY` to set and how to write pass the `{provider_name}/{model_name}` to the constructor.

        Some examples:

        - `anyscale/mistralai/Mistral-7B-Instruct-v0.1`, with `ANYSCALE_API_KEY`
        - `together_ai/togethercomputer/llama-2-70b-chat`, with `TOGETHERAI_API_KEY`
        - `sagemaker/<your-endpoint-name>`, with `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `AWS_REGION_NAME`
        - `azure/<your_deployment_name>`, with `AZURE_API_KEY`, `AZURE_API_BASE`, `AZURE_API_VERSION`, and the optional `AZURE_AD_TOKEN` and `AZURE_API_TYPE`

        
        If your provider offers an OpenAI-compatible endpoint, just add an `openai/` prefix to your full model name.

        ```python linenums="1"
        import dspy
        lm = dspy.LM('openai/your-model-name', api_key='PROVIDER_API_KEY', api_base='YOUR_PROVIDER_URL')
        dspy.configure(lm=lm)
        ```

## Calling the LM directly.

It's easy to call the `lm` you configured above directly. This gives you a unified API and lets you benefit from utilities like automatic caching.

```python linenums="1"       
lm("Say this is a test!", temperature=0.7)  # => ['This is a test!']
lm(messages=[{"role": "user", "content": "Say this is a test!"}])  # => ['This is a test!']
``` 

## Using the LM with DSPy modules.

Idiomatic DSPy involves using _modules_, which we discuss in the next guide.

```python linenums="1" 
# Define a module (ChainOfThought) and assign it a signature (return an answer, given a question).
qa = dspy.ChainOfThought('question -> answer')

# Run with the default LM configured with `dspy.configure` above.
response = qa(question="How many floors are in the castle David Gregory inherited?")
print(response.answer)
```
**Possible Output:**
```text
The castle David Gregory inherited has 7 floors.
```

## Using multiple LMs.

You can change the default LM globally with `dspy.configure` or change it inside a block of code with `dspy.context`.

!!! tip
    Using `dspy.configure` and `dspy.context` is thread-safe!


```python linenums="1" 
dspy.configure(lm=dspy.LM('openai/gpt-4o-mini'))
response = qa(question="How many floors are in the castle David Gregory inherited?")
print('GPT-4o-mini:', response.answer)

with dspy.context(lm=dspy.LM('openai/gpt-3.5-turbo')):
    response = qa(question="How many floors are in the castle David Gregory inherited?")
    print('GPT-3.5-turbo:', response.answer)
```
**Possible Output:**
```text
GPT-4o: The number of floors in the castle David Gregory inherited cannot be determined with the information provided.
GPT-3.5-turbo: The castle David Gregory inherited has 7 floors.
```

## Configuring LM generation.

For any LM, you can configure any of the following attributes at initialization or in each subsequent call.

```python linenums="1" 
gpt_4o_mini = dspy.LM('openai/gpt-4o-mini', temperature=0.9, max_tokens=3000, stop=None, cache=False)
```

By default LMs in DSPy are cached. If you repeat the same call, you will get the same outputs. But you can turn off caching by setting `cache=False`.


## Inspecting output and usage metadata.

Every LM object maintains the history of its interactions, including inputs, outputs, token usage (and $$$ cost), and metadata.

```python linenums="1" 
len(lm.history)  # e.g., 3 calls to the LM

lm.history[-1].keys()  # access the last call to the LM, with all metadata
```

**Output:**
```text
dict_keys(['prompt', 'messages', 'kwargs', 'response', 'outputs', 'usage', 'cost'])
```

### Advanced: Building custom LMs and writing your own Adapters.

Though rarely needed, you can write custom LMs by inheriting from `dspy.BaseLM`. Another advanced layer in the DSPy ecosystem is that of _adapters_, which sit between DSPy signatures and LMs. A future version of this guide will discuss these advanced features, though you likely don't need them.



## End of learn/programming/language_models.md

---


## File: learn/evaluation/overview.md

---
sidebar_position: 1
---

# Evaluation in DSPy

Once you have an initial system, it's time to **collect an initial development set** so you can refine it more systematically. Even 20 input examples of your task can be useful, though 200 goes a long way. Depending on your _metric_, you either just need inputs and no labels at all, or you need inputs and the _final_ outputs of your system. (You almost never need labels for the intermediate steps in your program in DSPy.) You can probably find datasets that are adjacent to your task on, say, HuggingFace datasets or in a naturally occuring source like StackExchange. If there's data whose licenses are permissive enough, we suggest you use them. Otherwise, you can label a few examples by hand or start deploying a demo of your system and collect initial data that way.

Next, you should **define your DSPy metric**. What makes outputs from your system good or bad? Invest in defining metrics and improving them incrementally over time; it's hard to consistently improve what you aren't able to define. A metric is a function that takes examples from your data and takes the output of your system, and returns a score. For simple tasks, this could be just "accuracy", e.g. for simple classification or short-form QA tasks. For most applications, your system will produce long-form outputs, so your metric will be a smaller DSPy program that checks multiple properties of the output. Getting this right on the first try is unlikely: start with something simple and iterate.

Now that you have some data and a metric, run development evaluations on your pipeline designs to understand their tradeoffs. Look at the outputs and the metric scores. This will probably allow you to spot any major issues, and it will define a baseline for your next steps.


??? "If your metric is itself a DSPy program..."
    If your metric is itself a DSPy program, a powerful way to iterate is to optimize your metric itself. That's usually easy because the output of the metric is usually a simple value (e.g., a score out of 5), so the metric's metric is easy to define and optimize by collecting a few examples.



## End of learn/evaluation/overview.md

---


## File: learn/evaluation/data.md

---
sidebar_position: 5
---

# Data

DSPy is a machine learning framework, so working in it involves training sets, development sets, and test sets. For each example in your data, we distinguish typically between three types of values: the inputs, the intermediate labels, and the final label. You can use DSPy effectively without any intermediate or final labels, but you will need at least a few example inputs.


## DSPy `Example` objects

The core data type for data in DSPy is `Example`. You will use **Examples** to represent items in your training set and test set. 

DSPy **Examples** are similar to Python `dict`s but have a few useful utilities. Your DSPy modules will return values of the type `Prediction`, which is a special sub-class of `Example`.

When you use DSPy, you will do a lot of evaluation and optimization runs. Your individual datapoints will be of type `Example`:

```python
qa_pair = dspy.Example(question="This is a question?", answer="This is an answer.")

print(qa_pair)
print(qa_pair.question)
print(qa_pair.answer)
```
**Output:**
```text
Example({'question': 'This is a question?', 'answer': 'This is an answer.'}) (input_keys=None)
This is a question?
This is an answer.
```

Examples can have any field keys and any value types, though usually values are strings.

```text
object = Example(field1=value1, field2=value2, field3=value3, ...)
```

You can now express your training set for example as:

```python
trainset = [dspy.Example(report="LONG REPORT 1", summary="short summary 1"), ...]
```


### Specifying Input Keys

In traditional ML, there are separated "inputs" and "labels".

In DSPy, the `Example` objects have a `with_inputs()` method, which can mark specific fields as inputs. (The rest are just metadata or labels.)

```python
# Single Input.
print(qa_pair.with_inputs("question"))

# Multiple Inputs; be careful about marking your labels as inputs unless you mean it.
print(qa_pair.with_inputs("question", "answer"))
```

Values can be accessed using the `.`(dot) operator. You can access the value of key `name` in defined object `Example(name="John Doe", job="sleep")` through `object.name`. 

To access or exclude certain keys, use `inputs()` and `labels()` methods to return new Example objects containing only input or non-input keys, respectively.

```python
article_summary = dspy.Example(article= "This is an article.", summary= "This is a summary.").with_inputs("article")

input_key_only = article_summary.inputs()
non_input_key_only = article_summary.labels()

print("Example object with Input fields only:", input_key_only)
print("Example object with Non-Input fields only:", non_input_key_only)
```

**Output**
```
Example object with Input fields only: Example({'article': 'This is an article.'}) (input_keys=None)
Example object with Non-Input fields only: Example({'summary': 'This is a summary.'}) (input_keys=None)
```

<!-- ## Loading Dataset from sources

One of the most convenient way to import datasets in DSPy is by using `DataLoader`. The first step is to declare an object, this object can then be used to call utilities to load datasets in different formats:

```python
from dspy.datasets import DataLoader

dl = DataLoader()
```

For most dataset formats, it's quite straightforward you pass the file path to the corresponding method of the format and you'll get the list of `Example` for the dataset in return:

```python
import pandas as pd

csv_dataset = dl.from_csv(
    "sample_dataset.csv",
    fields=("instruction", "context", "response"),
    input_keys=("instruction", "context")
)

json_dataset = dl.from_json(
    "sample_dataset.json",
    fields=("instruction", "context", "response"),
    input_keys=("instruction", "context")
)

parquet_dataset = dl.from_parquet(
    "sample_dataset.parquet",
    fields=("instruction", "context", "response"),
    input_keys=("instruction", "context")
)

pandas_dataset = dl.from_pandas(
    pd.read_csv("sample_dataset.csv"),    # DataFrame
    fields=("instruction", "context", "response"),
    input_keys=("instruction", "context")
)
```

These are some formats that `DataLoader` supports to load from file directly. In the backend, most of these methods leverage the `load_dataset` method from `datasets` library to load these formats. But when working with text data you often use HuggingFace datasets, in order to import HF datasets in list of `Example` format we can use `from_huggingface` method:

```python
blog_alpaca = dl.from_huggingface(
    "intertwine-expel/expel-blog",
    input_keys=("title",)
)
```

You can access the splits of the dataset by accessing the corresponding key:

```python
train_split = blog_alpaca['train']

# Since this is the only split in the dataset we can split this into 
# train and test split ourselves by slicing or sampling 75 rows from the train
# split for testing.
testset = train_split[:75]
trainset = train_split[75:]
```

The way you load a HuggingFace dataset using `load_dataset` is exactly how you load data it via `from_huggingface` as well. This includes passing specific splits, subsplits, read instructions, etc. For code snippets, you can refer to the [cheatsheet snippets](/cheatsheet/#dspy-dataloaders) for loading from HF. -->

## End of learn/evaluation/data.md

---


## File: learn/evaluation/metrics.md

---
sidebar_position: 5
---

# Metrics

DSPy is a machine learning framework, so you must think about your **automatic metrics** for evaluation (to track your progress) and optimization (so DSPy can make your programs more effective).


## What is a metric and how do I define a metric for my task?

A metric is just a function that will take examples from your data and the output of your system and return a score that quantifies how good the output is. What makes outputs from your system good or bad? 

For simple tasks, this could be just "accuracy" or "exact match" or "F1 score". This may be the case for simple classification or short-form QA tasks.

However, for most applications, your system will output long-form outputs. There, your metric should probably be a smaller DSPy program that checks multiple properties of the output (quite possibly using AI feedback from LMs).

Getting this right on the first try is unlikely, but you should start with something simple and iterate. 


## Simple metrics

A DSPy metric is just a function in Python that takes `example` (e.g., from your training or dev set) and the output `pred` from your DSPy program, and outputs a `float` (or `int` or `bool`) score.

Your metric should also accept an optional third argument called `trace`. You can ignore this for a moment, but it will enable some powerful tricks if you want to use your metric for optimization.

Here's a simple example of a metric that's comparing `example.answer` and `pred.answer`. This particular metric will return a `bool`.

```python
def validate_answer(example, pred, trace=None):
    return example.answer.lower() == pred.answer.lower()
```

Some people find these utilities (built-in) convenient:

- `dspy.evaluate.metrics.answer_exact_match`
- `dspy.evaluate.metrics.answer_passage_match`

Your metrics could be more complex, e.g. check for multiple properties. The metric below will return a `float` if `trace is None` (i.e., if it's used for evaluation or optimization), and will return a `bool` otherwise (i.e., if it's used to bootstrap demonstrations).

```python
def validate_context_and_answer(example, pred, trace=None):
    # check the gold label and the predicted answer are the same
    answer_match = example.answer.lower() == pred.answer.lower()

    # check the predicted answer comes from one of the retrieved contexts
    context_match = any((pred.answer.lower() in c) for c in pred.context)

    if trace is None: # if we're doing evaluation or optimization
        return (answer_match + context_match) / 2.0
    else: # if we're doing bootstrapping, i.e. self-generating good demonstrations of each step
        return answer_match and context_match
```

Defining a good metric is an iterative process, so doing some initial evaluations and looking at your data and outputs is key.


## Evaluation

Once you have a metric, you can run evaluations in a simple Python loop.

```python
scores = []
for x in devset:
    pred = program(**x.inputs())
    score = metric(x, pred)
    scores.append(score)
```

If you need some utilities, you can also use the built-in `Evaluate` utility. It can help with things like parallel evaluation (multiple threads) or showing you a sample of inputs/outputs and the metric scores.

```python
from dspy.evaluate import Evaluate

# Set up the evaluator, which can be re-used in your code.
evaluator = Evaluate(devset=YOUR_DEVSET, num_threads=1, display_progress=True, display_table=5)

# Launch evaluation.
evaluator(YOUR_PROGRAM, metric=YOUR_METRIC)
```


## Intermediate: Using AI feedback for your metric

For most applications, your system will output long-form outputs, so your metric should check multiple dimensions of the output using AI feedback from LMs.

This simple signature could come in handy.

```python
# Define the signature for automatic assessments.
class Assess(dspy.Signature):
    """Assess the quality of a tweet along the specified dimension."""

    assessed_text = dspy.InputField()
    assessment_question = dspy.InputField()
    assessment_answer: bool = dspy.OutputField()
```

For example, below is a simple metric that checks a generated tweet (1) answers a given question correctly and (2) whether it's also engaging. We also check that (3) `len(tweet) <= 280` characters.

```python
def metric(gold, pred, trace=None):
    question, answer, tweet = gold.question, gold.answer, pred.output

    engaging = "Does the assessed text make for a self-contained, engaging tweet?"
    correct = f"The text should answer `{question}` with `{answer}`. Does the assessed text contain this answer?"
    
    correct =  dspy.Predict(Assess)(assessed_text=tweet, assessment_question=correct)
    engaging = dspy.Predict(Assess)(assessed_text=tweet, assessment_question=engaging)

    correct, engaging = [m.assessment_answer for m in [correct, engaging]]
    score = (correct + engaging) if correct and (len(tweet) <= 280) else 0

    if trace is not None: return score >= 2
    return score / 2.0
```

When compiling, `trace is not None`, and we want to be strict about judging things, so we will only return `True` if `score >= 2`. Otherwise, we return a score out of 1.0 (i.e., `score / 2.0`).


## Advanced: Using a DSPy program as your metric

If your metric is itself a DSPy program, one of the most powerful ways to iterate is to compile (optimize) your metric itself. That's usually easy because the output of the metric is usually a simple value (e.g., a score out of 5) so the metric's metric is easy to define and optimize by collecting a few examples.



### Advanced: Accessing the `trace`

When your metric is used during evaluation runs, DSPy will not try to track the steps of your program.

But during compiling (optimization), DSPy will trace your LM calls. The trace will contain inputs/outputs to each DSPy predictor and you can leverage that to validate intermediate steps for optimization.


```python
def validate_hops(example, pred, trace=None):
    hops = [example.question] + [outputs.query for *_, outputs in trace if 'query' in outputs]

    if max([len(h) for h in hops]) > 100: return False
    if any(dspy.evaluate.answer_exact_match_str(hops[idx], hops[:idx], frac=0.8) for idx in range(2, len(hops))): return False

    return True
```


## End of learn/evaluation/metrics.md

---


## File: deep-dive/optimizers/Ensemble.md

---
sidebar_position: 4
---

# teleprompt.Ensemble

!!! warning "This page is outdated and may not be fully accurate in DSPy 2.5"


### Constructor

The constructor initializes the `Ensemble` class and sets up its attributes. This teleprompter is designed to create ensembled versions of multiple programs, reducing various outputs from different programs into a single output.

```python
class Ensemble(Teleprompter):
    def __init__(self, *, reduce_fn=None, size=None, deterministic=False):
```

**Parameters:**
- `reduce_fn` (_callable_, _optional_): Function used to reduce multiple outputs from different programs into a single output. A common choice is `dspy.majority`. Defaults to `None`.
- `size` (_int_, _optional_): Number of programs to randomly select for ensembling. If not specified, all programs will be used. Defaults to `None`.
- `deterministic` (_bool_, _optional_): Specifies whether ensemble should operate deterministically. Currently, setting this to `True` will raise an error as this feature is pending implementation. Defaults to `False`.

### Method

#### `compile(self, programs)`

This method compiles an ensemble of programs into a single program that when run, can either randomly sample a subset of the given programs to produce outputs or use all of them. The multiple outputs can then be reduced into a single output using the `reduce_fn`.

**Parameters:**
- `programs` (_list_): List of programs to be ensembled.

**Returns:**
- `EnsembledProgram` (_Module_): An ensembled version of the input programs.

### Example

```python
import dspy
from dspy.teleprompt import Ensemble

# Assume a list of programs
programs = [program1, program2, program3, ...]

# Define Ensemble teleprompter
teleprompter = Ensemble(reduce_fn=dspy.majority, size=2)

# Compile to get the EnsembledProgram
ensembled_program = teleprompter.compile(programs)
```

## End of deep-dive/optimizers/Ensemble.md

---


## File: deep-dive/optimizers/miprov2.md

---
sidebar_position: 6
---

# `MIPROv2` Optimizer

!!! warning "This page is outdated and may not be fully accurate in DSPy 2.5"


## Overview

`MIPROv2` (<u>M</u>ultiprompt <u>I</u>nstruction <u>PR</u>oposal <u>O</u>ptimizer Version 2) is an prompt optimizer capable of optimizing both instructions and few-shot examples jointly. It does this by bootstrapping few-shot example candidates, proposing instructions grounded in different dynamics of the task, and finding an optimized combination of these options using Bayesian Optimization. It can be used for optimizing few-shot examples & instructions jointly, or just instructions for 0-shot optimization.

## Example Usage

### Setting up a Sample Pipeline

We'll be making a basic answer generation pipeline over the GSM8K dataset. So let's start by configuring the LM which will be OpenAI LM client with `gpt-3.5-turbo` as the LLM in use.

```python
import dspy

turbo = dspy.OpenAI(model='gpt-3.5-turbo', max_tokens=250)
dspy.settings.configure(lm=turbo)
```

Now that we have the LM client setup it's time to import the train-dev split in `GSM8k` class that DSPy provides us:

```python
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric

gsm8k = GSM8K()

trainset, devset = gsm8k.train, gms8k.dev
```

We'll now define a basic QA inline signature i.e. `question->answer` and pass it to `ChainOfThought` module, that applies necessary addition for CoT style prompting to the Signature.

```python
class CoT(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought("question -> answer")
    
    def forward(self, question):
        return self.prog(question=question)
```

Now we need to evaluate this pipeline too! So we'll use the `Evaluate` class that DSPy provides us, as for the metric we'll use the `gsm8k_metric` that we imported above.

```python
from dspy.evaluate import Evaluate

evaluate = Evaluate(devset=devset[:], metric=gsm8k_metric, num_threads=8, display_progress=True, display_table=False)
```

To evaluate the `CoT` pipeline we'll need to create an object of it and pass it as an arg to the `evaluator` call.

```python
program = CoT()

evaluate(program, devset=devset[:])
```

Now we have the baseline pipeline ready to use, so let's try using the `MIPROv2` optimizer to improve our pipeline's performance!

### Optimizing with `MIPROv2`
To get started with `MIPROv2`, we'd recommend using the `auto` flag, starting with a `light` optimization run. This will set up hyperparameters for you to do a light optimization run on your program.

```python
# Import the optimizer
from dspy.teleprompt import MIPROv2

# Initialize optimizer
teleprompter = MIPROv2(
    metric=gsm8k_metric,
    auto="light", # Can choose between light, medium, and heavy optimization runs
)

# Optimize program
print(f"Optimizing program with MIPRO...")
optimized_program = teleprompter.compile(
    program.deepcopy(),
    trainset=trainset,
    max_bootstrapped_demos=3,
    max_labeled_demos=4,
    requires_permission_to_run=False,
)

# Save optimize program for future use
optimized_program.save(f"mipro_optimized")

# Evaluate optimized program
print(f"Evaluate optimized program...")
evaluate(optimized_program, devset=devset[:])
```

#### Optimizing instructions only with `MIPROv2` (0-Shot)

In some cases, we may want to only optimize the instruction, rather than including few-shot examples in the prompt. The code below demonstrates how this can be done using `MIPROv2`. Note that the key difference involves setting `max_labeled_demos` and `max_bootstrapped_demos` to zero.
```python
# Import the optimizer
from dspy.teleprompt import MIPROv2

# Initialize optimizer
teleprompter = MIPROv2(
    metric=gsm8k_metric,
    auto="light", # Can choose between light, medium, and heavy optimization runs
)

# Optimize program
print(f"Optimizing zero-shot program with MIPRO...")
zeroshot_optimized_program = teleprompter.compile(
    program.deepcopy(),
    trainset=trainset,
    max_bootstrapped_demos=0, # ZERO FEW-SHOT EXAMPLES
    max_labeled_demos=0, # ZERO FEW-SHOT EXAMPLES
    requires_permission_to_run=False,
)

# Save optimize program for future use
zeroshot_optimized_program.save(f"mipro_zeroshot_optimized")

# Evaluate optimized program
print(f"Evaluate optimized program...")
evaluate(zeroshot_optimized_program, devset=devset[:])
```

#### Optimizing with `MIPROv2` (advanced)
Once you've gotten a feel for using `MIPROv2` with `auto` settings, you may want to experiment with setting hyperparameters yourself to get the best results. The code below shows an example of how you can go about this. A full description of each parameter can be found in the section below.

```python
# Import the optimizer
from dspy.teleprompt import MIPROv2

# Initialize optimizer
teleprompter = MIPROv2(
    metric=gsm8k_metric,
    num_candidates=7,
    init_temperature=0.5,
    max_bootstrapped_demos=3,
    max_labeled_demos=4,
    verbose=False,
)

# Optimize program
print(f"Optimizing program with MIPRO...")
optimized_program = teleprompter.compile(
    program.deepcopy(),
    trainset=trainset,
    num_trials=15,
    minibatch_size=25,
    minibatch_full_eval_steps=10,
    minibatch=True, 
    requires_permission_to_run=False,
)

# Save optimize program for future use
optimized_program.save(f"mipro_optimized")

# Evaluate optimized program
print(f"Evaluate optimized program...")
evaluate(optimized_program, devset=devset[:])
```

## Parameters

### Initialization Parameters

| Parameter            | Type         | Default                                              | Description                                                                                                                  |
|----------------------|--------------|------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------|
| `metric`           | `dspy.metric`       | **Required**                                                 | The evaluation metric used to optimize the task model.                                                                       |
| `prompt_model`     | `dspy.LM`       | LM specified in `dspy.settings`                        | Model used for prompt generation.                                                                                            |
| `task_model`       | `dspy.LM`       | LM specified in `dspy.settings`                        | Model used for task execution.                                                                                               |
| `auto`       | `Optional[str]`       | None                        | If set to `light`, `medium`, or `heavy`, this will automatically configure the following hyperparameters: `num_candidates`, `num_trials`, `minibatch`, and will also cap the size of `valset` up to 100, 300, and 1000 for `light`, `medium`, and `heavy` runs respectively.                                                            |
| `num_candidates`   | `int`      | `10`                                                   | Number of candidate instructions & few-shot examples to generate and evaluate for each predictor. If `num_candidates=10`, this means for a 2 module LM program we'll be optimizing over 10 candidates x 2 modules x 2 variables (few-shot ex. and instructions for each module)= 40 total variables. Therefore, if we increase `num_candidates`, we will probably want to increase `num_trials` as well (see Compile parameters).                                                                          |
| `num_threads`      | `int`      |  `6`                                       | Threads to use for evaluation.                                                                                               |
| `max_errors`       | `int`      | `10`                                                 | Maximum errors during an evaluation run that can be made before throwing an Exception.                                       |
| `teacher_settings` | `dict`       | `{}`                                                 | Settings to use for the teacher model that bootstraps few-shot examples. An example dict would be `{lm=<dspy.LM object>}`. If your LM program with your default model is struggling to bootstrap any examples, it could be worth using a more powerful teacher model for bootstrapping.                                                     |
| `max_bootstrapped_demos` | `int`  | `4`      | Maximum number of bootstrapped demonstrations to generate and include in the prompt.                              |
| `max_labeled_demos`      | `int`  | `16`      | Maximum number of labeled demonstrations to generate and include in the prompt. Note that these differ from bootstrapped examples because they are just inputs & outputs sampled directly from the training set and do not have bootstrapped intermediate steps. 
| `init_temperature` | `float`        | `1.0`                                                  | The initial temperature for prompt generation, influencing creativity.                                                       |
| `verbose`          | `bool`      | `False`                                                | Enables printing intermediate steps and information.                                                                         |
| `track_stats`      | `bool`      | `True`                                                | Logs relevant information through the optimization process if set to True.                                                                       |
| `metric_threshold` | `float`        | `None`                                                 | A metric threshold is used if we only want to keep bootstrapped few-shot examples that exceed some threshold of performance.  |
| `seed` | `int`        | `9`                                                 | Seed for reproducibility.  |

### Compile Parameters

| Parameter                  | Type     | Default | Description                                                                                              |
|----------------------------|----------|---------|----------------------------------------------------------------------------------------------------------|
| `student`                | `dspy.Module`   | **Required**    | The base program to optimize.                                                                            |
| `trainset`               | `List[dspy.Example]`  | **Required** | Training dataset which is used to bootstrap few-shot examples and instructions. If a separate `valset` is not specified, 80% of this training set will also be used as a validation set for evaluating new candidate prompts.                                                                    |
| `valset`               | `List[dspy.Example]`  | Defaults to 80% of trainset | Dataset which is used to evaluate candidate prompts. We recommend using somewhere between 50-500 examples for optimization.                                                                      |
| `teacher`                | `dspy.Module`   | Defaults to student    | The program to run in order to bootstrap the few-shot examples.                                                                            |
| `num_trials`            | `int`  | `30`      | Number of optimization trials to run. When `minibatch` is set to `True`, this represents the number of minibatch trials that will be run on batches of size `minibatch_size`. When minibatch is set to `False`, each trial uses a full evaluation on the training set. In both cases, we recommend setting `num_trials` to a *minimum* of .75 x # modules in program x # variables per module (2 if few-shot examples & instructions will both be optimized, 1 in the 0-shot case).        |
| `minibatch`              | `bool`  | `True`    | Flag to enable evaluating over minibatches of data (instead of the full validation set) for evaluation each trial.                                                           |
| `minibatch_size`         | `int`  | `25.0`    | Size of minibatches for evaluations.                                                                     |
| `minibatch_full_eval_steps` | `int` | `10`    | When minibatching is enabled, a full evaluation on the validation set will be carried out every `minibatch_full_eval_steps` on the top averaging set of prompts (according to their average score on the minibatch trials).          
| `max_bootstrapped_demos` | `Optional[int]`  | Defaults to `init` value. | Maximum number of bootstrapped demonstrations to generate and include in the prompt.                              |
| `max_labeled_demos`      | `Optional[int]`  | Defaults to `init` value.  | Maximum number of labeled demonstrations to generate and include in the prompt. Note that these differ from bootstrapped examples because they are just inputs & outputs sampled directly from the training set and do not have bootstrapped intermediate steps.                             |
| `seed`                   | `Optional[int]`  | Defaults to `init` value. | Seed for reproducibility.                                                    |                                  |
| `program_aware_proposer` | `bool`  | `True`    | Flag to enable summarizing a reflexive view of the code for your LM program.                             |
| `data_aware_proposer` | `bool`  | `True`    | Flag to enable summarizing your training dataset.                             |
| `view_data_batch_size` | `int`  | `10`    | Number of data examples to look at a time when generating the summary.                             |
| `tip_aware_proposer` | `bool`  | `True`    | Flag to enable using a randomly selected tip for instruction generation.                             |
| `fewshot_aware_proposer` | `bool`  | `True`    | Flag to enable using generated few-shot examples for instruction proposal.                             |
| `requires_permission_to_run` | `bool` | `True`  | Flag to require user confirmation before running optimizations.                                          |

## How `MIPROv2` works

At a high level, `MIPROv2` works by creating both few-shot examples and new instructions for each predictor in your LM program, and then searching over these using Bayesian Optimization to find the best combination of these variables for your program.

These steps are broken down in more detail below:
1) **Bootstrap Few-Shot Examples**: The same bootstrapping technique used in `BootstrapFewshotWithRandomSearch` is used to create few-shot examples. This works by randomly sampling examples from your training set, which are then run through your LM program. If the output from the program is correct for this example, it is kept as a valid few-shot example candidate. Otherwise, we try another example until we've curated the specified amount of few-shot example candidates. This step creates `num_candidates` sets of `max_bootstrapped_demos` bootstrapped examples and `max_labeled_demos` basic examples sampled from the training set.
2) **Propose Instruction Candidates**. Next, we propose instruction candidates for each predictor in the program. This is done using another LM program as a proposer, which bootstraps & summarizes relevant information about the task to generate high quality instructions. Specifically, the instruction proposer includes (1) a generated summary of properties of the training dataset, (2) a generated summary of your LM program's code and the specific predictor that an instruction is being generated for, (3) the previously bootstrapped few-shot examples to show reference inputs / outputs for a given predictor and (4) a randomly sampled tip for generation (i.e. "be creative", "be concise", etc.) to help explore the feature space of potential instructions.
3. **Find an Optimized Combination of Few-Shot Examples & Instructions**. Finally, now that we've created these few-shot examples and instructions, we use Bayesian Optimization to choose which set of these would work best for each predictor in our program. This works by running a series of `num_trials` trials, where a new set of prompts are evaluated over our validation set at each trial. This helps the Bayesian Optimizer learn which combination of prompts work best over time. If `minibatch` is set to `True` (which it is by default), then the new set of prompts are only evaluated on a minibatch of size `minibatch_size` at each trial which generally allows for more efficient exploration / exploitation. The best averaging set of prompts is then evaluated on the full validation set every `minibatch_full_eval_steps` get a less noisey performance benchmark. At the end of the optimization process, the LM program with the set of prompts that performed best on the full validation set is returned.


For those interested in more details, more information on `MIPROv2` along with a study on `MIPROv2` compared with other DSPy optimizers can be found in [this paper](https://arxiv.org/abs/2406.11695). 


## End of deep-dive/optimizers/miprov2.md

---


## File: deep-dive/optimizers/bfrs.md

---
sidebar_position: 3
---

# BootstrapFewShotWithRandomSearch

!!! warning "This page is outdated and may not be fully accurate in DSPy 2.5"


`BootstrapFewShotWithRandomSearch` is an teleprompter that extends `BootstrapFewShot` by incorporating a random search strategy to optimize the selection of few-shot examples. This teleprompter is useful when you want to explore a variety of example combinations to find the best examples for your model.

## Usage

In terms of API `BootstrapFewShotWithRandomSearch` teleprompter is quite similar to `BootstrapFewShot` with few additional parameters needed. More precisely the parameters are:

- `metric` (callable): The metric function used to evaluate examples during bootstrapping and final evaluation.
- `teacher_settings` (dict, optional): Settings for the teacher predictor. Defaults to an empty dictionary.
- `max_bootstrapped_demos` (int, optional): Maximum number of bootstrapped demonstrations per predictor. Defaults to 4.
- `max_labeled_demos` (int, optional): Maximum number of labeled demonstrations per predictor. Defaults to 16.
- `max_rounds` (int, optional): Maximum number of bootstrapping rounds. Defaults to 1.
- `num_candidate_programs` (int): Number of candidate programs to generate during random search. Defaults to 16.
- `num_threads` (int): Number of threads used for evaluation during random search. Defaults to 6.
- `max_errors` (int): Maximum errors permitted during evaluation. Halts run with the latest error message. Defaults to 10.
- `stop_at_score` (float, optional): Score threshold for random search to stop early. Defaults to None.
- `metric_threshold` (float, optional): Score threshold for the metric to determine a successful example. Defaults to None.

## Working Example

Let's take the example of optimizing a simple CoT pipeline for GSM8k dataset, we'll take the example in [BootstrapFewShot](/deep-dive/optimizers/bootstrap-fewshot) as our running example for optimizers. We're gonna assume our data and pipeline is same as the on in `BootstrapFewShot` article. So let's start by initializing the optimizer:

```python
from dspy.teleprompt import BootstrapFewShotWithRandomSearch

teleprompter = BootstrapFewShotWithRandomSearch(
    metric=gsm8k_metric, 
    max_bootstrapped_demos=8, 
    max_labeled_demos=8,
    num_threads=10,
    num_candidate_programs=10
)
```

Now that we have initialized our teleprompter, let's compile our CoT module with the call to `compile` standard to all optimizers:

```python
cot_compiled = teleprompter.compile(CoT(), trainset=trainset, valset=devset)
```

Once the training is done you'll have a more optimized module that you can save or load again for use anytime:

```python
cot_compiled.save('turbo_gsm8k.json')

# Loading:
# cot = CoT()
# cot.load('turbo_gsm8k.json')
```

## End of deep-dive/optimizers/bfrs.md

---


## File: deep-dive/optimizers/LabeledFewShot.md

---
sidebar_position: 2
---

# LabeledFewShot

!!! warning "This page is outdated and may not be fully accurate in DSPy 2.5"


### Constructor

The constructor initializes the `LabeledFewShot` class and sets up its attributes, particularly defining `k` number of samples to be used by the predictor.

```python
class LabeledFewShot(Teleprompter):
    def __init__(self, k=16):
        self.k = k
```

**Parameters:**
- `k` (_int_): Number of samples to be used for each predictor. Defaults to 16.

### Method

#### `compile(self, student, *, trainset)`

This method compiles the `LabeledFewShot` instance by configuring the `student` predictor. It assigns subsets of the `trainset` in each student's predictor's `demos` attribute. If the `trainset` is empty, the method returns the original `student`.

**Parameters:**
- `student` (_Teleprompter_): Student predictor to be compiled.
- `trainset` (_list_): Training dataset for compiling with student predictor.

**Returns:**
- The compiled `student` predictor with assigned training samples for each predictor or the original `student` if the `trainset` is empty.

### Example

```python
import dspy

#Assume defined trainset
class RAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()

        #declare retrieval and predictor modules
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
    
    #flow for answering questions using predictor and retrieval modules
    def forward(self, question):
        context = self.retrieve(question).passages
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)

#Define teleprompter
teleprompter = LabeledFewShot()

# Compile!
compiled_rag = teleprompter.compile(student=RAG(), trainset=trainset)
```

## End of deep-dive/optimizers/LabeledFewShot.md

---


## File: deep-dive/optimizers/BootstrapFinetune.md

---
sidebar_position: 4
---

# teleprompt.BootstrapFinetune

!!! warning "This page is outdated and may not be fully accurate in DSPy 2.5"


### Constructor

### `__init__(self, metric=None, teacher_settings={}, multitask=True)`

The constructor initializes a `BootstrapFinetune` instance and sets up its attributes. It defines the teleprompter as a `BootstrapFewShot` instance for the finetuning compilation.

```python
class BootstrapFinetune(Teleprompter):
    def __init__(self, metric=None, teacher_settings={}, multitask=True):
```

**Parameters:**
- `metric` (_callable_, _optional_): Metric function to evaluate examples during bootstrapping. Defaults to `None`.
- `teacher_settings` (_dict_, _optional_): Settings for teacher predictor. Defaults to empty dictionary.
- `multitask` (_bool_, _optional_): Enable multitask fine-tuning. Defaults to `True`.

### Method

#### `compile(self, student, *, teacher=None, trainset, valset=None, target='t5-large', bsize=12, accumsteps=1, lr=5e-5, epochs=1, bf16=False)`

This method first compiles for bootstrapping with the `BootstrapFewShot` teleprompter. It then prepares fine-tuning data by generating prompt-completion pairs for training and performs finetuning. After compilation, the LMs are set to the finetuned models and the method returns a compiled and fine-tuned predictor.

**Parameters:**
- `student` (_Predict_): Student predictor to be fine-tuned.
- `teacher` (_Predict_, _optional_): Teacher predictor to help with fine-tuning. Defaults to `None`.
- `trainset` (_list_): Training dataset for fine-tuning.
- `valset` (_list_, _optional_): Validation dataset for fine-tuning. Defaults to `None`.
- `target` (_str_, _optional_): Target model for fine-tuning. Defaults to `'t5-large'`.
- `bsize` (_int_, _optional_): Batch size for training. Defaults to `12`.
- `accumsteps` (_int_, _optional_): Gradient accumulation steps. Defaults to `1`.
- `lr` (_float_, _optional_): Learning rate for fine-tuning. Defaults to `5e-5`.
- `epochs` (_int_, _optional_): Number of training epochs. Defaults to `1`.
- `bf16` (_bool_, _optional_): Enable mixed-precision training with BF16. Defaults to `False`.

**Returns:**
- `compiled2` (_Predict_): A compiled and fine-tuned `Predict` instance.

### Example

```python
#Assume defined trainset
#Assume defined RAG class
...

#Define teleprompter
teleprompter = BootstrapFinetune(teacher_settings=dict({'lm': teacher}))

# Compile!
compiled_rag = teleprompter.compile(student=RAG(), trainset=trainset, target='google/flan-t5-base')
```

## End of deep-dive/optimizers/BootstrapFinetune.md

---


## File: deep-dive/optimizers/bootstrap-fewshot.md

---
sidebar_position: 1
---

# BootstrapFewShot

!!! warning "This page is outdated and may not be fully accurate in DSPy 2.5"


When compiling a DSPy program, we generally invoke an optimizer that takes the program, a training set, and a metric and returns a new optimized program. Different optimizers apply different strategies for optimization. This family of optimizers is focused on optimizing the few shot examples. Let's take an example of a Sample pipeline and see how we can use this optimizer to optimize it.

## Setting up a Sample Pipeline

We will be making a basic answer generation pipeline over the GSM8K dataset. We won't be changing anything in it! So let's start by configuring the LM which will be OpenAI LM client with `gpt-3.5-turbo` as the LLM in use.

```python
import dspy

turbo = dspy.LM(model='openai/gpt-3.5-turbo', max_tokens=250)
dspy.configure(lm=turbo)
```

Now that we have the LM client setup it's time to import the train-dev split in `GSM8k` class that DSPy provides us:

```python
import random
from dspy.datasets import DataLoader

dl = DataLoader()

gsm8k = dl.from_huggingface(
    "gsm8k",
    "main",
    input_keys = ("question",),
)

trainset, devset = gsm8k['train'], random.sample(gsm8k['test'], 50)
```

We will now define a basic QA inline signature i.e. `question->answer` and pass it to `ChainOfThought` module, that applies necessary addition for CoT style prompting to the Signature.

```python
class CoT(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought("question -> answer")

    def forward(self, question):
        return self.prog(question=question)
```

Now we need to evaluate this pipeline too!! So we'll use the `Evaluate` class that DSPy provides us, as for the metric we'll use the `gsm8k_metric` that we imported above.

```python
from dspy.evaluate import Evaluate

evaluate = Evaluate(devset=devset[:], metric=gsm8k_metric, num_threads=NUM_THREADS, display_progress=True, display_table=False)
```

To evaluate the `CoT` pipeline we'll need to create an object of it and pass it as an arg to the `evaluator` call.

```python
cot_baseline = CoT()

evaluate(cot_baseline)
```

Now we have the baseline pipeline ready to use, so let's try using the `BootstrapFewShot` optimizer and optimizing our pipeline to make it even better!

## Using `BootstrapFewShot`

Let's start by importing and initializing our optimizer, for the metric we'll be using the same `gsm8k_metric` imported and used above:

```python
from dspy.teleprompt import BootstrapFewShot

optimizer = BootstrapFewShot(
    metric=gsm8k_metric,
    max_bootstrapped_demos=8,
    max_labeled_demos=8,
    max_rounds=10,
)
```

`metric` refers to the metric that we want to optimize the pipeline for.

`max_rounds` refers to the maximum number of rounds of training we want to perform. During the bootstrapping process we train the student on the teacher's predictions and then use the teacher to generate new examples to add to the training set. This process is repeated for `max_rounds` times and for each round, the `temperature` parameter of the teacher is increased from a baseline value of `0.7` to `0.7 + 0.001 * round_number`.

What are `max_bootstrapped_demos` and `max_labeled_demos` parameters? Let's see the differences via a table:

| Feature                   | `max_labeled_demos`                                                                                                                                                                                                                        | `max_bootstrapped_demos`                                                                                                                                                                                                                                                                                                    |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Purpose**               | Refers to the maximum number of labeled demonstrations (examples) that will be used for training the student module directly. Labeled demonstrations are typically pre-existing, manually labeled examples that the module can learn from. | Refers to the maximum number of demonstrations that will be bootstrapped. Bootstrapping in this context means generating new training examples based on the predictions of the teacher module or some other process. These bootstrapped demonstrations are then used alongside or instead of the manually labeled examples. |
| **Training Usage**        | Directly used in training; typically more reliable due to manual labeling.                                                                                                                                                                 | Augment training data; potentially less accurate as they are generated examples.                                                                                                                                                                                                                                            |
| **Data Source**           | Pre-existing dataset of manually labeled examples.                                                                                                                                                                                         | Generated during the training process, often using outputs from a teacher module.                                                                                                                                                                                                                                           |
| **Influence on Training** | Higher quality and reliability, assuming labels are accurate.                                                                                                                                                                              | Provides more data but may introduce noise or inaccuracies.                                                                                                                                                                                                                                                                 |

This optimizer augments any necessary field even if you data doesn't have it, for example we don't have rationale for labelling however you'll see the rationale for each few shot example in the prompt that this optimizer curated, how? By generating them all via a `teacher` module which is an optional parameter, since we didn't pass it the optimizer creates a teacher from the module we are training or the `student` module.

In the next section, we'll seen this process step by step but for now let's start optimizing our `CoT` module by calling the `compile` method in the optimizer:

```python
cot_compiled = optimizer.compile(CoT(), trainset=trainset)
```

Once the training is done you'll have a more optimized module that you can save or load again for use anytime:

```python
cot_compiled.save('turbo_gsm8k.json')

# Loading:
# cot = CoT()
# cot.load('turbo_gsm8k.json')
```

## How `BootstrapFewShot` works?

`LabeledFewShot` is the most vanilla optimizer that takes a training set as input and it assigns subsets of the trainset in each student's predictor's demos attribute. You can think of it as basically adding few shot examples to the prompt.

`BootStrapFewShot` follows the following steps:

1. Initializing a student program which is the one we are optimizing and a teacher program which unless specified otherwise is a clone of the student.

2. Using the `LabeledFewShot` optimizer it adds the labeled demonstrations to the teacher's predictor's `demos` attribute. The `LabeledFewShot` optimizer randomly samples `max_labeled_demos` from the trainset.

3. Mappings are created between the names of predictors and their corresponding instances in both student and teacher models.

4. Then it bootstraps the teacher by generating `max_bootstrapped_demos` examples using the inputs from the trainset and the teacher's predictions.

5. The process iterates over each example in the training set. For each example, the method checks if the maximum number of bootstraps has been reached. If so, the process stops.

6. For each training example, the teacher model attempts to generate a prediction.

7. If the teacher model successfully generates a prediction that satisfies the metric, the trace of this prediction process is captured. This trace includes details about which predictors were called, the inputs they received, and the outputs they produced.

8. If the prediction is successful, a demonstration (demo) is created for each step in the trace. This demo includes the inputs to the predictor and the outputs it generated.

9. Other optimizers like `BootstrapFewShotWithOptuna`, `BootstrapFewShotWithRandomSearch` etc. also work on same principles with slight changes in the example discovery process.



## End of deep-dive/optimizers/bootstrap-fewshot.md

---


## File: deep-dive/optimizers/copro.md

---
sidebar_position: 5
---

# `COPRO`

!!! warning "This page is outdated and may not be fully accurate in DSPy 2.5"


`COPRO` which aims to improve the output prefixes and instruction of the signatures in a module in a zero/few shot setting. This teleprompter is especially beneficial for fine-tuning the prompt for language models and ensure they perform tasks more effectively, all from a vague and unrefined prompt.

## Setting up a Sample Pipeline

We'll be creating our CoT pipeline from scratch including the metric itself! So let's start by configuring the LM which will be OpenAI LM client with `gpt-3.5-turbo` as the LLM in use.

```python
import dspy

turbo = dspy.OpenAI(model='gpt-3.5-turbo')
dspy.settings.configure(lm=turbo)
```

Now that we have the LM client setup it's time to import the train-dev split in `HotPotQA` class that DSPy provides us:

```python
from dspy.datasets import HotPotQA

dataset = HotPotQA(train_seed=1, train_size=20, eval_seed=2023, dev_size=50, test_size=0)

trainset, devset = dataset.train, dataset.dev
```

We'll now define a class based signature for QA task similar to `question->answer` and pass it to `ChainOfThought` module, that will give us the result via Chain Of Thought from the LM client for this signature.

```python
class CoTSignature(dspy.Signature):
    """Answer the question and give the reasoning for the same."""

    question = dspy.InputField(desc="question about something")
    answer = dspy.OutputField(desc="often between 1 and 5 words")

class CoTPipeline(dspy.Module):
    def __init__(self):
        super().__init__()

        self.signature = CoTSignature
        self.predictor = dspy.ChainOfThought(self.signature)

    def forward(self, question):
        result = self.predictor(question=question)
        return dspy.Prediction(
            answer=result.answer,
            reasoning=result.reasoning,
        )
```

Now we need to evaluate this pipeline too!! So we'll use the `Evaluate` class that DSPy provides us, as for the metric we'll use the `validate_context_and_answer` that we'll define. `validate_context_and_answer` uses `dspy.evaluate.answer_exact_match` metric in DSPy which in essence sees if pred and example are same or not.

```python
from dspy.evaluate import Evaluate

def validate_context_and_answer(example, pred, trace=None):
    answer_EM = dspy.evaluate.answer_exact_match(example, pred)
    return answer_EM

NUM_THREADS = 5
evaluate = Evaluate(devset=devset, metric=validate_context_and_answer, num_threads=NUM_THREADS, display_progress=True, display_table=False)
```

To evaluate the `CoTPipeline` we'll need to create an object of it and pass it as an arg to the `evaluator` call.

```python
cot_baseline = CoTPipeline()

devset_with_input = [dspy.Example({"question": r["question"], "answer": r["answer"]}).with_inputs("question") for r in devset]
evaluate(cot_baseline, devset=devset_with_input)
```

Now we have the baseline pipeline ready to use, so let's try using the `COPRO` teleprompter and optimizing our pipeline to make it even better!

## Using `COPRO`

Let's start by importing and initializing our teleprompter, for the metric we'll be using the same `validate_context_and_answer` imported and used above:

```python
from dspy.teleprompt import COPRO

teleprompter = COPRO(
    metric=validate_context_and_answer,
    verbose=True,
)
```

In this teleprompter there is a breadth and depth argument that defines the number of instruction/prefix candidate and number of iterations in the optimization step. We'll understand this in depth in the next section. This teleprompter comes up with better instruction candidates for the signature and better prefix candidates for the output fields of the signature. Let's start optimizing our `CoT` module by calling the `compile` method in the teleprompter:

```python
kwargs = dict(num_threads=64, display_progress=True, display_table=0) # Used in Evaluate class in the optimization process

compiled_prompt_opt = teleprompter.compile(cot, trainset=trainset, eval_kwargs=kwargs)
```

Once the training is done you'll have better instructions and prefixes that you'll need to edit in signature manually. So let's say the output during optimization is like:

```text
i: "Please answer the question and provide your reasoning for the answer. Your response should be clear and detailed, explaining the rationale behind your decision. Please ensure that your answer is well-reasoned and supported by relevant explanations and examples."
p: "[Rationale]"
Average Metric (78.9) ...
```

Then you'll copy this and edit the original instruction class to:

```python
class CoTSignature(dspy.Signature):
    """Please answer the question and provide your reasoning for the answer. Your response should be clear and detailed, explaining the rationale behind your decision. Please ensure that your answer is well-reasoned and supported by relevant explanations and examples."""

    question = dspy.InputField(desc="question about something")
    reasoning = dspy.OutputField(desc="reasoning for the answer", prefix="[Rationale]")
    answer = dspy.OutputField(desc="often between 1 and 5 words")
```

!!! info
    The prefix would be proposed only for the output field that is defined first i.e. reasoning in `CoTSignature`.

Reinitialize the Pipeline object and reevaluate the pipeline! And now you have a more powerful predictor with more optimized Signature!

## How `COPRO` works?

It is interesting that to get optimal prefixes and instruction, `COPRO` uses Signatures. Basically `COPRO` uses Signature to optimize Signature!! Let's look at the codebase a bit more closely:

```python
class BasicGenerateInstruction(Signature):
    """You are an instruction optimizer for large language models. I will give you a ``signature`` of fields (inputs and outputs) in English. Your task is to propose an instruction that will lead a good language model to perform the task well. Don't be afraid to be creative."""

    basic_instruction = dspy.InputField(desc="The initial instructions before optimization")
    proposed_instruction = dspy.OutputField(desc="The improved instructions for the language model")
    proposed_prefix_for_output_field = dspy.OutputField(desc="The string at the end of the prompt, which will help the model start solving the task")

class GenerateInstructionGivenAttempts(dspy.Signature):
    """You are an instruction optimizer for large language models. I will give some task instructions I've tried, along with their corresponding validation scores. The instructions are arranged in increasing order based on their scores, where higher scores indicate better quality.

Your task is to propose a new instruction that will lead a good language model to perform the task even better. Don't be afraid to be creative."""

    attempted_instructions = dspy.InputField(format=dsp.passages2text)
    proposed_instruction = dspy.OutputField(desc="The improved instructions for the language model")
    proposed_prefix_for_output_field = dspy.OutputField(desc="The string at the end of the prompt, which will help the model start solving the task")
```

These two signatures are what give use the optimal instruction and prefixes. Now, the `BasicGenerateInstruction` will generate `n` instruction and prefixes based on the `breadth` parameter, basically `n=breadth`. This happens only one time in the start to seed the instruction attempts.

It uses these instructions and pass them to `GenerateInstructionGivenAttempts` which outputs hopefully a more optimal instruction. This then happens for `m` iterations which is the `depth` parameter in DSPy.

![Signature Optimizer](./img/signature_optimizer_process_v4.png)

Let's break down the process stepwise:

1. **Starting Point:** Use BasicGenerateInstruction to create initial optimized instructions and prefixes. This is based on a basic instruction input.
2. **Iterative Improvement:** Pass these initial instructions to GenerateInstructionGivenAttempts.
3. **Repeat Optimization:** In each iteration (up to m times):
    - Evaluate the current instructions and their effectiveness.
    - Propose new, more optimized instructions and prefixes based on the evaluation.
4. **Outcome:** After m iterations, the system ideally converges to a set of highly optimized instructions and corresponding prefixes that lead to better performance of the language model on the given task.

This iterative approach allows for continuous refinement of instructions and prefixes, leveraging the strengths of the teleprompter and improving task performance over time.

***


## End of deep-dive/optimizers/copro.md

---


## File: deep-dive/retrieval_models_clients/ColBERTv2.md

import AuthorDetails from '@site/src/components/AuthorDetails';

# ColBERTv2

## Setting up the ColBERTv2 Client

The constructor initializes the `ColBERTv2` class instance and sets up the request parameters for interacting with the ColBERTv2 retrieval server. This server is hosted remotely at `'http://20.102.90.50:2017/wiki17_abstracts`. 

- `url` (_str_): URL for ColBERTv2 server. Defaults to `"http://0.0.0.0"`
- `port` (_Union[str, int]_, _Optional_): Port endpoint for ColBERTv2 server. Defaults to `None`.
- `post_requests` (_bool_, _Optional_): Flag for using HTTP POST requests. Defaults to `False`.

Example of the ColBERTv2 constructor:

```python
class ColBERTv2:
    def __init__(
        self,
        url: str = "http://0.0.0.0",
        port: Optional[Union[str, int]] = None,
        post_requests: bool = False,
    ):
```

## Under the Hood

### `__call__(self, query: str, k: int = 10, simplify: bool = False) -> Union[list[str], list[dotdict]]`

**Parameters:**
- `query` (_str_): Search query string used for retrieval sent to ColBERTv2 server.
- `k` (_int_, _optional_): Number of passages to retrieve. Defaults to 10.
- `simplify` (_bool_, _optional_): Flag for simplifying output to a list of strings. Defaults to False.

**Returns:**
- `Union[list[str], list[dotdict]]`: Depending on `simplify` flag, either a list of strings representing the passage content (`True`) or a list of `dotdict` instances containing passage details (`False`).

Internally, the method handles the specifics of preparing the request query to the ColBERTv2 server and corresponding payload to obtain the response. 

The function handles the retrieval of the top-k passages based on the provided query.

If `post_requests` is set, the method sends a query to the server via a POST request else via a GET request.

It then processes and returns the top-k passages from the response with the list of retrieved passages dependent on the `simplify` flag return condition above.


## Sending Retrieval Requests via ColBERTv2 Client
1) _**Recommended**_ Configure default RM using `dspy.configure`.

This allows you to define programs in DSPy and have DSPy internally conduct retrieval using `dsp.retrieve` on the query on the configured RM.

```python
import dspy
import dsp

dspy.settings.configure(rm=colbertv2_wiki17_abstracts)
retrieval_response = dsp.retrieve("When was the first FIFA World Cup held?", k=5)

for result in retrieval_response:
    print("Text:", result, "\n")
```


2) Generate responses using the client directly.
```python
import dspy

retrieval_response = colbertv2_wiki17_abstracts('When was the first FIFA World Cup held?', k=5)

for result in retrieval_response:
    print("Text:", result['text'], "\n")
```

***

<AuthorDetails name="Arnav Singhvi"/>

## End of deep-dive/retrieval_models_clients/ColBERTv2.md

---


## File: deep-dive/retrieval_models_clients/WatsonDiscovery.md

# WatsonDiscoveryRM

### Constructor

The constructor initializes the `WatsonDiscoveryRM` class instance and sets up the request parameters for interacting with Watson Discovery service at IBM Cloud.

```python
class WatsonDiscoveryRM:
    def __init__(
        self,
        apikey: str,
        url:str,
        version:str,
        project_id:str,
        collection_ids:list=[],
        k: int = 7,
    ):
```

**Parameters:**

- `apikey` (str): apikey for authentication purposes,
- `url` (str): endpoint URL that includes the service instance ID
- `version` (str): Release date of the version of the API you want to use. Specify dates in YYYY-MM-DD format.
- `project_id` (str): The Universally Unique Identifier (UUID) of the project.
- `collection_ids` (list): An array containing the collections on which the search will be executed.
- `k` (int, optional): The number of top passages to retrieve. Defaults to 7.

### Methods

#### `forward(self, query_or_queries: Union[str, list[str]], k: Optional[int]= None) -> dspy.Prediction:`

Search the Watson Discovery collection for the top `k` passages matching the given query or queries.

**Parameters:**

- `query_or_queries` (_Union[str, list[str]]_): The query or list of queries to search for.
- `k` (_Optional[int]_, _optional_): The number of results to retrieve. If not specified, defaults to the value set during initialization.

**Returns:**

- `dspy.Prediction`: Contains the retrieved passages, each represented as a `dotdict` with schema `[{"title":str, "long_text": str, "passage_score": float, "document_id": str, "collection_id": str, "start_offset": int, "end_offset": int, "field": str}]`

### Quickstart

```python
import dspy

retriever_model = WatsonDiscoveryRM(
    apikey = "Your API Key",
    url = "URL of the Watson Discovery Service",
    version = "2023-03-31",
    project_id = "Project Id",
    collection_ids = ["Collection ID"],
    k = 5
)

retrieval_response = retriever_model("Explore the significance of quantum computing",k=5)

for result in retrieval_response:
    print("Document:", result.long_text, "\n")
```


## End of deep-dive/retrieval_models_clients/WatsonDiscovery.md

---


## File: deep-dive/retrieval_models_clients/ClarifaiRM.md

# ClarifaiRM

[Clarifai](https://clarifai.com/) is a powerful AI platform that provides vector search capabilities through its search API. DSPy has integrated ClarifaiRM to support efficient text search and retrieval through its specialized indexing and ability to handle large-scale document collections.

To support passage retrieval, ClarifaiRM assumes that documents have been properly ingested into a Clarifai application with the following:
- Text data properly indexed and stored
- Appropriate search configurations set up in the Clarifai platform
- Valid authentication credentials (PAT key) with appropriate permissions

The ClarifaiRM module requires the `clarifai` Python package. If not already installed, you can install it using:

```bash
pip install clarifai
```

**Note:** 

Before using ClarifaiRM, ensure you have:

1. Created a Clarifai account and application
2. Ingested your documents into the application
3. Obtained your User ID, App ID, and Personal Access Token (PAT)

## Setting up the ClarifaiRM Client

The constructor initializes an instance of the `ClarifaiRM` class, which requires authentication credentials and configuration to connect to your Clarifai application.

- `clarifai_user_id` (_str_): Your unique Clarifai user identifier.
- `clarifai_app_id` (_str_): The ID of your Clarifai application where documents are stored.
- `clarifai_pat` (_Optional[str]_): Your Clarifai Personal Access Token (PAT). It will look for `CLARIFAI_PAT` in environment variables if not provided.
- `k` (_int_, _optional_): The number of top passages to retrieve. Defaults to 3.

Example of the ClarifaiRM constructor:

```python
ClarifaiRM(
    clarifai_user_id: str,
    clarifai_app_id: str,
    clarifai_pat: Optional[str] = None,
    k: int = 3,
)
```

**Note:** 

The PAT can be provided either directly to the constructor or through the `CLARIFAI_PAT` environment variable. For security best practices, using environment variables is recommended.

## Under the Hood

### `retrieve_hits(self, hits)`

**Parameters:**
- `hits` (_ClarifaiHit_): A hit object from Clarifai's search response.

**Returns:**
- `str`: The retrieved text content.

Internal method that retrieves text content from the hit's URL using authenticated requests.

### `forward(self, query_or_queries: Union[str, List[str]], k: Optional[int] = None, **kwargs) -> dspy.Prediction`

**Parameters:**
- `query_or_queries` (_Union[str, List[str]]_): The query or list of queries to search for.
- `k` (_Optional[int]_, _optional_): The number of results to retrieve. If not specified, defaults to the value set during initialization.
- `**kwargs`: Additional keyword arguments passed to Clarifai's search function.

**Returns:**
- `dspy.Prediction`: Contains the retrieved passages, each represented as a `dotdict` with a `long_text` attribute.

Search the Clarifai application for the top `k` passages matching the given query or queries. Uses parallel processing with ThreadPoolExecutor to efficiently retrieve multiple results.

## Examples

### Basic Usage
```python
import os
from dspy.retrieve.clarifai_rm import ClarifaiRM
import dspy

os.environ["CLARIFAI_PAT"] = "your_pat_key"

retriever_model = ClarifaiRM(
    clarifai_user_id="your_user_id",
    clarifai_app_id="your_app_id",
    k=5
)

turbo = dspy.OpenAI(model="gpt-3.5-turbo")
dspy.settings.configure(lm=turbo, rm=retriever_model)

results = retriever_model("Explore the significance of quantum computing")
```

### Multiple Queries
```python
queries = [
    "What is machine learning?",
    "How does deep learning work?",
    "Explain neural networks"
]

results = retriever_model(queries, k=3)
```

### Using with DSPy Retrieve Module
```python
from dspy import Retrieve

retrieve = Retrieve(k=5)

class RAG(dspy.Module):
    def __init__(self):
        super().__init__()
        self.retrieve = Retrieve(k=3)
    
    def forward(self, query):
        passages = self.retrieve(query)
        return passages

rag = RAG()
result = rag("What are the latest developments in AI?")
```

### Handling Results
```python
results = retriever_model("quantum computing advances", k=5)

for i, result in enumerate(results, 1):
    print(f"Result {i}:")
    print(result.long_text)
    print("-" * 50)

first_passage = results[0].long_text

num_results = len(results)
```

### Integration with Other DSPy Components
```python
from dspy import ChainOfThought, Predict, Retrieve 

# Create a simple QA chain
class QAChain(dspy.Module):
    def __init__(self):
        super().__init__()
        self.retrieve = Retrieve(k=3)
        self.generate_answer = ChainOfThought("question, context -> answer")
    
    def forward(self, question):
        context = self.retrieve(question)
        answer = self.generate_answer(question=question, context=context)
        return answer

qa = QAChain()
answer = qa("What are the main applications of quantum computing?")
```

### Error Handling Example
```python
try:
    retriever_model = ClarifaiRM(
        clarifai_user_id="your_user_id",
        clarifai_app_id="your_app_id",
        clarifai_pat="invalid_pat"
    )
    results = retriever_model("test query")
except Exception as e:
    print(f"Error occurred: {e}")
```

**Note:** 

These examples assume you have:

- A properly configured Clarifai application
- Valid authentication credentials
- Documents already ingested into your Clarifai app
- The necessary environment variables set up

## End of deep-dive/retrieval_models_clients/ClarifaiRM.md

---


## File: deep-dive/retrieval_models_clients/MyScaleRM.md

# MyScaleRM

## Constructor

Initializes an instance of the `MyScaleRM` class, which is designed to use MyScaleDB (a ClickHouse fork optimized for vector similarity and full-text search) to retrieve documents based on query embeddings. This class supports embedding generation using either local models or OpenAI's API and manages database interactions efficiently.

### Syntax

```python
MyScaleRM(
    client: clickhouse_connect.driver.client.Client,
    table: str,
    database: str = 'default',
    metadata_columns: List[str] = ['text'],
    vector_column: str = 'vector',
    k: int = 3,
    openai_api_key: Optional[str] = None,
    openai_model: Optional[str] = None,
    local_embed_model: Optional[str] = None
)
```

## Parameters for `MyScaleRM` Constructor

- `client` (_clickhouse_connect.driver.client.Client_): A client connection to the MyScaleDB database, used to execute queries and manage interactions with the database.
- `table` (_str_): Specifies the table within MyScaleDB from which data will be retrieved. This table should be equipped with a vector column for conducting similarity searches.
- `database` (_str_, optional): The name of the database where the table is located, defaulting to `"default"`.
- `metadata_columns` (_List[str], optional_): Columns to include as metadata in the output, defaulting to `["text"]`.
- `vector_column` (_str, optional_): The column that contains vector data, used for similarity searches, defaulting to `"vector"`.
- `k` (_int, optional_): The number of closest matches to return for each query, defaulting to 3.
- `openai_api_key` (_str, optional_): API key for accessing OpenAI services, necessary if using OpenAI for embedding generation.
- `openai_model` (_str, optional_): The specific OpenAI model to use for embeddings, required if an OpenAI API key is provided.
- `local_embed_model` (_str, optional_): Specifies a local model for embedding generation, chosen if local computation is preferred.

## Methods

### `forward`

Executes a retrieval operation based on a user's query and returns the top `k` relevant results using the embeddings generated by the specified method.

### Syntax

```python
def forward(self, user_query: str, k: Optional[int] = None) -> dspy.Prediction
```

## Parameters

- `user_query` (_str_): The query to retrieve matching passages.
- `k` (_Optional[int], optional_): The number of top matches to retrieve. If not provided, it defaults to the `k` value set during class initialization.

## Returns

- `dspy.Prediction`: Contains the retrieved passages, formatted as a list of `dotdict` objects. Each entry includes:
  - **long_text (str)**: The text content of the retrieved passage.

## Description

The `forward` method leverages the MyScaleDB's vector search capabilities to find the top `k` passages that best match the provided query. This method is integral for utilizing the MyScaleRM class to access and retrieve data efficiently based on semantic similarity, facilitated by the chosen embedding generation technique (either via a local model or the OpenAI API).

## Quickstart

This section provides practical examples of how to instantiate and use the `MyScaleRM` class to retrieve data from MyScaleDB efficiently using text embeddings.

```python
from dspy.retrieve.myscaledb_rm import MyScaleRM

MyScale_model = MyScaleRM(client=client,
               table="table_name",
               openai_api_key="sk-***",
               openai_model="embeddings_model",
               vector_column="vector_column_name",
               metadata_columns=["add_your_columns_here"],
               k=6)

MyScale_model("Please suggest me some funny movies")

passages = results.passages

# Loop through each passage and print the 'long_text'
for passage in passages:
    print(passage['long_text'], "\n")

```


## End of deep-dive/retrieval_models_clients/MyScaleRM.md

---


## File: deep-dive/retrieval_models_clients/ChromadbRM.md

# ChromadbRM

!!! note
    Adapted from documentation provided by https://github.com/animtel

ChromadbRM have the flexibility from a variety of embedding functions as outlined in the [chromadb embeddings documentation](https://docs.trychroma.com/embeddings). While different options are available, this example demonstrates how to utilize OpenAI embeddings specifically.


## Setting up the ChromadbRM Client

The constructor initializes an instance of the `ChromadbRM` class, with the option to use OpenAI's embeddings or any alternative supported by chromadb, as detailed in the official [chromadb embeddings documentation](https://docs.trychroma.com/guides/embeddings).

- `collection_name` (_str_): The name of the chromadb collection.
- `persist_directory` (_str_): Path to the directory where chromadb data is persisted.
- `embedding_function` (_Optional[EmbeddingFunction[Embeddable]]_, _optional_): The function used for embedding documents and queries. Defaults to `DefaultEmbeddingFunction()` if not specified.
- `k` (_int_, _optional_): The number of top passages to retrieve. Defaults to 7.

Example of the ChromadbRM constructor: 

```python
ChromadbRM(
    collection_name: str,
    persist_directory: str,
    embedding_function: Optional[EmbeddingFunction[Embeddable]] = OpenAIEmbeddingFunction(),
    k: int = 7,
)
```

## Under the Hood

### `forward(self, query_or_queries: Union[str, List[str]], k: Optional[int] = None) -> dspy.Prediction`

**Parameters:**
- `query_or_queries` (_Union[str, List[str]]_): The query or list of queries to search for.
- `k` (_Optional[int]_, _optional_): The number of results to retrieve. If not specified, defaults to the value set during initialization.

**Returns:**
- `dspy.Prediction`: Contains the retrieved passages, each represented as a `dotdict` with a `long_text` attribute.

Search the chromadb collection for the top `k` passages matching the given query or queries, using embeddings generated via the specified `embedding_function`.

## Sending Retrieval Requests via ChromadbRM Client

```python
from dspy.retrieve.chromadb_rm import ChromadbRM
import os
import openai
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

embedding_function = OpenAIEmbeddingFunction(
    api_key=os.environ.get('OPENAI_API_KEY'),
    model_name="text-embedding-ada-002"
)

retriever_model = ChromadbRM(
    'your_collection_name',
    '/path/to/your/db',
    embedding_function=embedding_function,
    k=5
)

results = retriever_model("Explore the significance of quantum computing", k=5)

for result in results:
    print("Document:", result.long_text, "\n")
```

## End of deep-dive/retrieval_models_clients/ChromadbRM.md

---


## File: deep-dive/retrieval_models_clients/custom-rm-client.md

# Creating Custom RM Client

DSPy provides support for various retrieval modules out of the box like `ColBERTv2`, `AzureCognitiveSearch`, `Lancedb`, `Pinecone`, `Weaviate`, etc. Unlike Language Model (LM) modules, creating a custom RM module is much more simple and flexible. 

As of now, DSPy offers 2 ways to create a custom RM: the Pythonic way and the DSPythonic way. We'll take a look at both, understand why both are performing the same behavior, and how you can implement each!

## I/O of RM Client

Before understanding the implementation, let's understand the idea and I/O within RM modules. 

The **input** to an RM module is either 1) a single query or 2) a list of queries.

The **output** is the `top-k` passages per query retrieved from a retrieval model, vector store, or search client. 

![I/O in RM Module](./img/io_rm_module.png)

Conventionally, we simply call the RM module object through the `__call__` method, inputting the query/queries as argument(s) of the call with the corresponding output returned as a list of strings. 

We'll see how this I/O is essentially same in both methods of implementation but differs in their delivery.

## The Pythonic Way

To account for our RM I/O, we create a class that conducts the retrieval logic, which we implement in the `__init__` and `__call__` methods:

```python
from typing import List, Union

class PythonicRMClient:
    def __init__(self):
        pass

    def __call__(self, query: Union[str, List[str]], k:int) -> Union[List[str], List[List[str]]]:
        pass
```

!!! info
    Don't worry about the extensive type-hinting above. `typing` is a package that provides type-definitions for function inputs and outputs.

    `Union` covers all possible types of the argument/output. So:
    * `Union[str, List[str]]`: Assigned to `query` to work with a single query string or a list of queries strings.
    * `Union[List[str], List[List[str]]]`: Assigned to the output of `__call__` to work with a single query string as a list or a list of multiple query string lists.

So let's start by implementing `PythonicRMClient` for a local retrieval model hosted on a API with endpoint being `/`. We'll start by implementing the `__init__` method, which simply initializes the class attributes, `url` and `port`, and attaches the port to the url if present.

```python
def __init__(self, url: str, port:int = None):
    self.url = f`{url}:{port}` if port else url
```

Now it's time to write the retrieval logic in `__call__` method:

```python
def __call__(self, query:str, k:int) -> List[str]:
    params = {"query": query, "k": k}
    response = requests.get(self.url, params=params)

    response = response.json()["retrieved_passages"]    # List of top-k passages
    return response
```

This serves to represent our API request call to retrieve our list of **top-k passages** which we return as the response. Let's bring it all together and see how our RM class looks like:

```python
from typing import List

class PythonicRMClient:
    def __init__(self, url: str, port:int = None):
        self.url = f`{url}:{port}` if port else url

    def __call__(self, query:str, k:int) -> List[str]:
        # Only accept single query input, feel free to modify it to support 

        params = {"query": query, "k": k}
        response = requests.get(self.url, params=params)

        response = response.json()["retrieved_passages"]    # List of top k passages
        return response
```

That's all!! This is the most basic way to implement a RM model and mirrors DSP-v1-hosted RM models like `ColBERTv2` and `AzureCognitiveSearch`. 

Now let's take a look at how we streamline this process in DSPy!

## The DSPythonic Way

The DSPythonic way mirrors the Pythonic way in maintaining the same input but now returning an object of `dspy.Prediction` class, the standard output format for any DSPy module as we've seen in previous docs. Additionally, this class would now inherit the `dspy.Retrieve` class to maintain state management within the DSPy library.

So let's implement `__init__` and `forward` method where our class's `__call__` is calling the `forward` method as is=:

```python
import dspy
from typing import List, Union, Optional

class DSPythonicRMClient(dspy.Retrieve):
    def __init__(self, k:int):
        pass

    def forward(self, query: Union[str, List[str]], k:Optional[str]) -> dspy.Prediction:
        pass
```

Unlike `PythonicRMClient`, we initialize `k` as part of the initialization call and the `forward` method will take query/queries as arguments and the `k` number of retrieved passages as an optional argument. `k` is used within the inherited `dspy.Retrieve` initialization when we call `super().__init__()`.

We'll be implementing `DSPythonicRMClient` for the same local retrieval model API we used above. We'll start by implementing the `__init__` method, which mirrors the `PythonicRMClient`.

```python
def __init__(self, url: str, port:int = None, k:int = 3):
    super().__init__(k=k)

    self.url = f`{url}:{port}` if port else url
```

We'll now implement the `forward` method, returning the output as a `dspy.Prediction` object under the `passage` attribute which is standard among all the RM modules. The call will default to the defined `self.k` argument unless overridden in this call. 

```python
def forward(self, query:str, k:Optional[int]) -> dspy.Prediction:
    params = {"query": query, "k": k if k else self.k}
    response = requests.get(self.url, params=params)

    response = response.json()["retrieved_passages"]    # List of top k passages
    return dspy.Prediction(
        passages=response
    )
```

Let's bring it all together and see how our RM class looks like:

```python
import dspy
from typing import List, Union, Optional

class DSPythonicRMClient(dspy.Retrieve):
    def __init__(self, url: str, port:int = None, k:int = 3):
        super().__init__(k=k)

        self.url = f`{url}:{port}` if port else url

    def forward(self, query_or_queries:str, k:Optional[int]) -> dspy.Prediction:
        params = {"query": query_or_queries, "k": k if k else self.k}
        response = requests.get(self.url, params=params)

        response = response.json()["retrieved_passages"]    # List of top k passages
        return dspy.Prediction(
            passages=response
        )
```

That's all!! This is the way to implement a custom RM model client within DSPy and how more recently-supported RM models like `QdrantRM`, `WeaviateRM`, `LancedbRM`, etc. are implemented in DSPy. 

Let's take a look at how we use these retrievers.

## Using Custom RM Models

DSPy offers two ways of using custom RM clients: Direct Method and using `dspy.Retrieve`.

### Direct Method

The most straightforward way to use your custom RM is by directly using its object within the DSPy Pipeline. 

Let's take a look at the following pseudocode of a DSPy Pipeline as an example:

```python
class DSPyPipeline(dspy.Module):
    def __init__(self):
        super().__init__()

        url = "http://0.0.0.0"
        port = 3000

        self.pythonic_rm = PythonicRMClient(url=url, port=port)
        self.dspythonic_rm = DSPythonicRMClient(url=url, port=port, k=3)

        ...

    def forward(self, *args):
        ...

        passages_from_pythonic_rm = self.pythonic_rm(query)
        passages_from_dspythonic_rm = self.dspythonic_rm(query).passages

        ...
```

This ensures you retrieve a list of passages from your RM client and can interact with the results within your forward pass in whichever way needed for your pipeline's purpose!

### Using `dspy.Retrieve`

This way is more experimental in essence, allowing you to maintain the same pipeline and experiment with different RMs. How? By configuring it!

```python
import dspy

lm = ...
url = "http://0.0.0.0"
port = 3000

# pythonic_rm = PythonicRMClient(url=url, port=port)
dspythonic_rm = DSPythonicRMClient(url=url, port=port, k=3)

dspy.settings.configure(lm=lm, rm=dspythonic_rm)
```

Now, in the pipeline, you just need to use `dspy.Retrieve` which will use this `rm` client to get the top-k passage for a given query!

```python
class DSPyPipeline(dspy.Module):
    def __init__(self):
        super().__init__()

        url = "http://0.0.0.0"
        port = 3000

        self.rm = dspy.Retrieve(k=3)
        ...

    def forward(self, *args):
        ...

        passages = self.rm(query)

        ...
```

Now if you'd like to use a different RM, you can just update the `rm` parameter via `dspy.settings.configure`.

!!! info "How `dspy.Retrieve` uses `rm`"
    When we call `dspy.Retrieve` the `__call__` method will execute the `forward` method as is. In `forward`, the top-k passages are received by the `dsp.retrieveEnsemble` method in [search.py](https://github.com/stanfordnlp/dspy/blob/main/dsp/primitives/search.py).

    If an `rm` is not initialized in `dsp.settings`, this would raise an error.


## End of deep-dive/retrieval_models_clients/custom-rm-client.md

---


## File: deep-dive/retrieval_models_clients/RAGatouilleRM.md

# RAGatouilleRM

### Constructor

The constructor initializes the `RAGatouille` class instance and sets up the required parameters for interacting with the index created using [RAGatouille library](https://github.com/bclavie/RAGatouille).

```python
class RAGatouilleRM(dspy.Retrieve):
    def __init__(
        self,
        index_root: str,
        index_name: str,
        k: int = 3,
    ):
```

**Parameters:**

- `index_root` (_str_): Folder path where your index is stored.
- `index_name` (_str_): Name of the index you want to retrieve from.
- `k` (_int_): The default number of passages to retrieve. Defaults to `3`.

### Methods

#### `forward(self, query_or_queries: Union[str, List[str]], k:Optional[int]) -> dspy.Prediction`

Enables making queries to the RAGatouille-made index for retrieval. Internally, the method handles the specifics of preparing the query to obtain the response. The function handles the retrieval of the top-k passages based on the provided query.

**Parameters:**

- `query_or_queries` (Union[str, List[str]]): Query string used for retrieval.
- `k` (_int_, _optional_): Number of passages to retrieve. Defaults to 3.

**Returns:**

- `dspy.Prediction`: List of k passages


## End of deep-dive/retrieval_models_clients/RAGatouilleRM.md

---


## File: deep-dive/retrieval_models_clients/YouRM.md

# YouRM

### Constructor

Initialize an instance of the `YouRM` class that calls the [You.com APIs](https://documentation.you.com/api-reference/search) for web-based document retrieval. Options available are the "Search" and "News" APIs.

```python
YouRM(
    ydc_api_key: Optional[str] = None,
    k: int = 3,
    endpoint: Literal["search", "news"] = "search",
    num_web_results: Optional[int] = None,
    safesearch: Optional[Literal["off", "moderate", "strict"]] = None,
    country: Optional[str] = None,
    search_lang: Optional[str] = None,
    ui_lang: Optional[str] = None,
    spellcheck: Optional[bool] = None,
)
```

**Parameters:**

- `ydc_api_key` (Optional[str]): you.com API key, if `YDC_API_KEY` is not set in the environment
- `k` (int): If `endpoint="search"`, the max snippets to return per search hit.
  If `endpoint="news"`, the max articles to return.
- `endpoint` (Literal["search", "news"]): you.com endpoints
- `num_web_results` (Optional[int]): The max number of web results to return, must be under 20
- `safesearch` (Optional[Literal["off", "moderate", "strict"]]): Safesearch settings, one of "off", "moderate", "strict", defaults to moderate
- `country` (Optional[str]): Country code, ex: 'US' for United States, see API reference for more info
- `search_lang` (Optional[str]): (News API) Language codes, ex: 'en' for English, see API reference for more info
- `ui_lang` (Optional[str]): (News API) User interface language for the response, ex: 'en' for English.
  See API reference for more info
- `spellcheck` (Optional[bool]): (News API) Whether to spell check query or not, defaults to True

### Methods

#### `forward(self, query_or_queries: Union[str, List[str]], k: Optional[int] = None) -> dspy.Prediction`

If `endpoint="search"`, search the web for the top `k` snippets matching the given query or queries.

If `endpoint="news"`, search the web for the top `k` articles matching the given query or queries.

**Parameters:**

- `query_or_queries` (_Union[str, List[str]]_): The query or list of queries to search for.
- `k` (_Optional[int]_, _optional_): The number of results to retrieve. If not specified, defaults to the value set during initialization.

**Returns:**

- `dspy.Prediction`: Contains the retrieved passages, each represented as a `dotdict` with schema `[{"long_text": str}]`

### Quickstart

Obtain a You.com API key from [https://api.you.com/](https://api.you.com/).

Export this key to an environment variable `YDC_API_KEY`.

```python
from dspy.retrieve.you_rm import YouRM
import os

# The retriever obtains the API key from the `YDC_API_KEY` env var
retriever_model = YouRM(endpoint="search")

results = retriever_model("Tell me about national parks in the US", k=5)

for result in results:
    print("Document:", result.long_text, "\n")
```


## End of deep-dive/retrieval_models_clients/YouRM.md

---


## File: deep-dive/retrieval_models_clients/MilvusRM.md

# MilvusRM

MilvusRM uses OpenAI's `text-embedding-3-small` embedding by default or any customized embedding function.
To support passage retrieval, it assumes that a Milvus collection has been created and populated with the following field:

- `text`: The text of the passage


## Set up the MilvusRM Client

The constructor initializes an instance of the `MilvusRM` class, with the option to use OpenAI's `text-embedding-3-small` embeddings or any customized embedding function .

- `collection_name (str)`: The name of the Milvus collection to query against.
- `uri (str, optional)`: The Milvus connection uri. Defaults to "http://localhost:19530".
- `token (str, optional)`: The Milvus connection token. Defaults to None.
- `db_name (str, optional)`: The Milvus database name. Defaults to "default".
- `embedding_function (callable, optional)`: The function to convert a list of text to embeddings.
    The embedding function should take a list of text strings as input and output a list of embeddings.
    Defaults to None. By default, it will get OpenAI client by the environment variable OPENAI_API_KEY and use OpenAI's embedding model "text-embedding-3-small" with the default dimension.
- `k (int, optional)`: The number of top passages to retrieve. Defaults to 3.

Example of the MilvusRM constructor: 

```python
MilvusRM(
    collection_name: str,
    uri: Optional[str] = "http://localhost:19530",
    token: Optional[str] = None,
    db_name: Optional[str] = "default",
    embedding_function: Optional[Callable] = None,
    k: int = 3,
)
```

## Under the Hood

### `forward(self, query_or_queries: Union[str, List[str]], k: Optional[int] = None) -> dspy.Prediction`

**Parameters:**
- `query_or_queries` (_Union[str, List[str]]_): The query or list of queries to search for.
- `k` (_Optional[int]_, _optional_): The number of results to retrieve. If not specified, defaults to the value set during initialization.

**Returns:**
- `dspy.Prediction`: Contains the retrieved passages, each represented as a `dotdict` with a `long_text` attribute.

Search the Milvus collection for the top `k` passages matching the given query or queries, using embeddings generated via the default OpenAI embedding or the specified `embedding_function`.

## Sending Retrieval Requests via MilvusRM Client

```python
from dspy.retrieve.milvus_rm import MilvusRM
import os

os.environ["OPENAI_API_KEY"] = "<YOUR_OPENAI_API_KEY>"

retriever_model = MilvusRM(
    collection_name="<YOUR_COLLECTION_NAME>",
    uri="<YOUR_MILVUS_URI>",
    token="<YOUR_MILVUS_TOKEN>"  # ignore this if no token is required for Milvus connection
    )

results = retriever_model("Explore the significance of quantum computing", k=5)

for result in results:
    print("Document:", result.long_text, "\n")
```

## End of deep-dive/retrieval_models_clients/MilvusRM.md

---


## File: deep-dive/retrieval_models_clients/Neo4jRM.md

# Neo4jRM

### Constructor

Initialize an instance of the `Neo4jRM` class.

```python
Neo4jRM(
    index_name: str,
    text_node_property: str,
    k: int = 5,
    retrieval_query: str = None,
    embedding_provider: str = "openai",
    embedding_model: str = "text-embedding-ada-002",
)
```

**Environment Variables:**

You need to define the credentials as environment variables:

- `NEO4J_USERNAME` (_str_): Specifies the username required for authenticating with the Neo4j database. This is a crucial security measure to ensure that only authorized users can access the database.

- `NEO4J_PASSWORD` (_str_): Defines the password associated with the `NEO4J_USERNAME` for authentication purposes. This password should be kept secure to prevent unauthorized access to the database.

- `NEO4J_URI` (_str_): Indicates the Uniform Resource Identifier (URI) used to connect to the Neo4j database. This URI typically includes the protocol, hostname, and port, providing the necessary information to establish a connection to the database.

- `NEO4J_DATABASE` (_str_, optional): Specifies the name of the database to connect to within the Neo4j instance. If not set, the system defaults to using `"neo4j"` as the database name. This allows for flexibility in connecting to different databases within a single Neo4j server.

- `OPENAI_API_KEY` (_str_): Specifies the API key required for authenticiating with OpenAI's services.

**Parameters:**

- `index_name` (_str_): Specifies the name of the vector index to be used within Neo4j for organizing and querying data.
- `text_node_property` (_str_, _optional_): Defines the specific property of nodes that will be returned.
- `k` (_int_, _optional_): The number of top results to return from the retrieval operation. It defaults to 5 if not explicitly specified.
- `retrieval_query` (_str_, _optional_): A custom query string provided for retrieving data. If not provided, a default query tailored to the `text_node_property` will be used.
- `embedding_provider` (_str_, _optional_): The name of the service provider for generating embeddings. Defaults to "openai" if not specified.
- `embedding_model` (_str_, _optional_): The specific embedding model to use from the provider. By default, it uses the "text-embedding-ada-002" model from OpenAI.

### Methods

#### `forward(self, query: [str], k: Optional[int] = None) -> dspy.Prediction`

Search the neo4j vector index for the top `k` passages matching the given query or queries, using embeddings generated via the specified `embedding_model`.

**Parameters:**

- `query` (str\_): The query.
- `k` (_Optional[int]_, _optional_): The number of results to retrieve. If not specified, defaults to the value set during initialization.

**Returns:**

- `dspy.Prediction`: Contains the retrieved passages as a list of string with the prediction signature.

ex:

```python
Prediction(
    passages=['Passage 1 Lorem Ipsum awesome', 'Passage 2 Lorem Ipsum Youppidoo', 'Passage 3 Lorem Ipsum Yassssss']
)
```

### Quick Example how to use Neo4j in a local environment.

```python
from dspy.retrieve.neo4j_rm import Neo4jRM
import os

os.environ["NEO4J_URI"] = 'bolt://localhost:7687'
os.environ["NEO4J_USERNAME"] = 'neo4j'
os.environ["NEO4J_PASSWORD"] = 'password'
os.environ["OPENAI_API_KEY"] = 'sk-'

retriever_model = Neo4jRM(
    index_name="vector",
    text_node_property="text"
)

results = retriever_model("Explore the significance of quantum computing", k=3)

for passage in results:
    print("Document:", passage, "\n")
```


## End of deep-dive/retrieval_models_clients/Neo4jRM.md

---


## File: deep-dive/retrieval_models_clients/WeaviateRM.md

# Weaviate Retrieval Model
[Weaviate](https://weaviate.io/) is an open-source vector database that can be used to retrieve relevant passages before passing it to the language model. Weaviate supports a variety of [embedding models](https://weaviate.io/developers/weaviate/model-providers) from OpenAI, Cohere, Google and more! Before building your DSPy program, you will need a Weaviate cluster running with data. You can follow this [notebook](https://github.com/weaviate/recipes/blob/main/integrations/llm-frameworks/dspy/Weaviate-Import.ipynb) as an example. 


## Configuring the Weaviate Client 
Weaviate is available via a hosted service ([WCD](https://console.weaviate.cloud/)) or as a self managed instance. You can learn about the different installation methods [here](https://weaviate.io/developers/weaviate/installation#installation-methods). 

* `weaviate_collection_name` (str): The name of the Weaviate collection
* `weaviate_client` (WeaviateClient): An instance of the Weaviate client
* `k` (int, optional): The number of top passages to retrieve. The default is set to `3`

An example of the WeaviateRM constructor: 

```python
WeaviateRM(
    weaviate_collection_name: str
    weaviate_client: str,
    k: int = 5
)
```

## Using Multitenancy
Multi-tenancy allows a collection to efficiently serve isolated groups of data. Each "tenant" in a multi-tenant collection can only access its own data, while sharing the same data structure and settings.

If your Weaviate instance is tenant-aware, you can provide a tenant_id in the WeaviateRM constructor or as a keyword argument:

```python
retriever_model = WeaviateRM(
    weaviate_collection_name="<WEAVIATE_COLLECTION>",
    weaviate_client=weaviate_client,
    tenant_id="tenant123"
)

results = retriever_model("Your query here", tenant_id="tenantXYZ")
```
When tenant_id is specified, this will scope all retrieval requests to the tenant ID provided.

## Under the Hood

`forward(self, query_or_queries: Union[str, List[str]], k: Optional[int] = None, **kwargs) -> dspy.Prediction`

### Parameters
* `query_or_queries` (Union[str, List[str]]): The query or queries to search for
* `k (Optional[int]): The number of top passages to retrieve. It defaults to `self.k`
*  `**kwargs`: Additional keyword arguments like `rerank` for example

### Returns
* `dspy.Prediction`: An object containing the retrieved passages


## Sending Retrieval Requests via the WeaviateRM Client

Here is an example of the Weaviate constructor using embedded:

```python
import weaviate
import dspy
from dspy.retrieve.weaviate_rm import WeaviateRM

weaviate_client = weaviate.connect_to_embedded() # you can also use local or WCD

retriever_model = WeaviateRM(
    weaviate_collection_name="<WEAVIATE_COLLECTION>",
    weaviate_client=weaviate_client 
)

results = retriever_model("Explore the significance of quantum computing", k=5)

for result in results:
    print("Document:", result.long_text, "\n")
```

You can follow along with more DSPy and Weaviate examples [here](https://weaviate.io/developers/integrations/llm-frameworks/dspy)!


## End of deep-dive/retrieval_models_clients/WeaviateRM.md

---


## File: deep-dive/retrieval_models_clients/FalkordbRM.md

# FalkordbRM

### Constructor

Initialize an instance of the `FalkordbRM` class.

```python
FalkordbRM(
    node_label: str,
    text_node_property: str,
    embedding_node_property: str,
    k: int = 5,
    retrieval_query: str,
    embedding_provider: str = "openai",
    embedding_model: str = "text-embedding-ada-002",
)
```

**Environment Variables:**

You need to define the credentials as environment variables

- `FALKORDB_HOST` (_str_): Specifies the host required for connecting with the Falkordb database. If not provided, the system will default to `localhost`

- `FALKORDB_PORT` (_int_): Specifies the port required for connecting with the Falkordb database. If not provided, the system will default to `6379`

- `FALKORDB_USERNAME` (_str_, optional): Specifies the username required for authenticating with a [Falkordb Cloud](https://app.falkordb.cloud/signin) database.

- `FALKORDB_PASSWORD` (_str_, optional): Specifies the password required for authenticating with a [Falkordb Cloud](https://app.falkordb.cloud/signin) database.

- `FALKORDB_DATABASE` (_str_, optional): Specifies the name of the database to connect to within the Falkordb instance. If not provided, the systems defaults to using a randomly generated four ascii_letters character string e.g "tari".

- `OPENAI_API_KEY` (_str_): Specifies the API key required for authenticating with OpenAI's services.

**Parameters:**

- `node_label` (_str_): Specifies the label of the node to be used within Falkordb for organizing and querying data.
- `text_node_property` (_str_, _optional_): Defines the specific text property of the node that will be returned.
- `embedding_node_property` (_str_): Defines the specific embedding property of the node that will be used within Falkordb for querying data.
- `k` (_int_, _optional_): The number of top results to return from the retrieval operation. It defaults to 5 if not explicitly specified.
- `retrieval_query` (_str_, _optional_): A custom query string provided for retrieving data. If not provided, a default query tailored to the `text_node_property` will be used.
- `embedding_provider` (_str_, _optional_): The name of the service provider for generating embeddings. Only "openai" is supported.
- `embedding_model` (_str_, _optional_): The specific embedding model to use from the provider. By default, it uses the "text-embedding-ada-002" model from OpenAI.


### Methods

#### `forward(self, query: [str], k: Optional[int] = None) -> dspy.Prediction`

Search the Falkordb vector index for the top `k` passages matching the given query or queries, using embeddings generated via the specified `embedding_model`.

**Parameters:**

- `query` (str\_): The query.
- `k` (_Optional[int]_, _optional_): The number of results to retrieve. If not specified, defaults to the value set during initialization.

**Returns:**

- `dspy.Prediction`: Contains the retrieved passages as a list of string with the prediction signature.

ex:

```python
Prediction(
    passages=['Passage 1 Lorem Ipsum awesom', 'Passage 2 Lorem Ipsum Youppidoo', 'Passage 3 Lorem Ipsum Yassssss']
)
```

### Quick Example how to use Falkordb in a local environment.

```python
from dspy.retrieve.falkordb_rm import FalkordbRM
import os


os.environ["FALKORDB_HOST"] = 'localhost'
os.environ["FALKORDB_PORT"] = 6379
os.environ["OPENAI_API_KEY"] = 'sk-'

retriever_model = FalkordbRM(
    node_label="myIndex",
    text_node_property="text",
    embedding_node_property="embedding"
)

results = retriever_model("Explore the significance of quantum computing", k=3)

for passage in results:
    print("Document:", passage, "\n")
```

## End of deep-dive/retrieval_models_clients/FalkordbRM.md

---


## File: deep-dive/retrieval_models_clients/FaissRM.md

# FaissRM

### Constructor

Initialize an instance of FaissRM by providing it with a vectorizer and a list of strings

```python
FaissRM(
    document_chunks: List[str],
    vectorizer: dsp.modules.sentence_vectorizer.BaseSentenceVectorizer,
    k: int = 3
)
```

**Parameters:**

- `document_chunks` (_List[str]_): a list of strings that comprises the corpus to search. You cannot add/insert/upsert to this list after creating this FaissRM object.
- `vectorizer` (_dsp.modules.sentence_vectorizer.BaseSentenceVectorizer_, _optional_): If not provided, a dsp.modules.sentence_vectorizer.SentenceTransformersVectorizer object is created and used.
- `k` (_int_, _optional_): The number of top passages to retrieve. Defaults to 3.

### Methods

#### `forward(self, query_or_queries: Union[str, List[str]]) -> dspy.Prediction`

Search the FaissRM vector database for the top `k` passages matching the given query or queries, using embeddings generated via the vectorizer specified at FaissRM construction time

**Parameters:**

- `query_or_queries` (_Union[str, List[str]]_): The query or list of queries to search for.

**Returns:**

- `dspy.Prediction`: Contains the retrieved passages, each represented as a `dotdict` with a `long_text` attribute and an `index` attribute. The `index` attribute is the index in the document_chunks array provided to this FaissRM object at construction time.

### Quickstart with the default vectorizer

The **FaissRM** module provides a retriever that uses an in-memory Faiss vector database. This module does not include a vectorizer; instead it supports any subclass of **dsp.modules.sentence_vectorizer.BaseSentenceVectorizer**. If a vectorizer is not provided, an instance of **dsp.modules.sentence_vectorizer.SentenceTransformersVectorizer** is created and used by **FaissRM**. Note that the default embedding model for **SentenceTransformersVectorizer** is **all-MiniLM-L6-v2**

```python
import dspy
from dspy.retrieve.faiss_rm import FaissRM

document_chunks = [
    "The superbowl this year was played between the San Francisco 49ers and the Kanasas City Chiefs",
    "Pop corn is often served in a bowl",
    "The Rice Bowl is a Chinese Restaurant located in the city of Tucson, Arizona",
    "Mars is the fourth planet in the Solar System",
    "An aquarium is a place where children can learn about marine life",
    "The capital of the United States is Washington, D.C",
    "Rock and Roll musicians are honored by being inducted in the Rock and Roll Hall of Fame",
    "Music albums were published on Long Play Records in the 70s and 80s",
    "Sichuan cuisine is a spicy cuisine from central China",
    "The interest rates for mortgages is considered to be very high in 2024",
]

frm = FaissRM(document_chunks)
turbo = dspy.OpenAI(model="gpt-3.5-turbo")
dspy.settings.configure(lm=turbo, rm=frm)
print(frm(["I am in the mood for Chinese food"]))
```


## End of deep-dive/retrieval_models_clients/FaissRM.md

---


## File: deep-dive/retrieval_models_clients/DatabricksRM.md

# DatabricksRM

### Constructor

Initialize an instance of the `DatabricksRM` retriever class, which enables DSPy programs to query
[Databricks Mosaic AI Vector Search](https://docs.databricks.com/en/generative-ai/vector-search.html#mosaic-ai-vector-search)
indexes for document retrieval.

```python
DatabricksRM(
    databricks_index_name: str,
    databricks_endpoint: Optional[str] = None,
    databricks_token: Optional[str] = None,
    columns: Optional[List[str]] = None,
    filters_json: Optional[str] = None,
    k: int = 3,
    docs_id_column_name: str = "id",
    text_column_name: str = "text",
)
```

**Parameters:**

- `databricks_index_name (str)`: The name of the Databricks Vector Search Index to query.
- `databricks_endpoint (Optional[str])`: The URL of the Databricks Workspace containing
  the Vector Search Index. Defaults to the value of the `DATABRICKS_HOST` environment variable.
  If unspecified, the Databricks SDK is used to identify the endpoint based on the current
  environment.
- `databricks_token (Optional[str])`: The Databricks Workspace authentication token to use
  when querying the Vector Search Index. Defaults to the value of the `DATABRICKS_TOKEN`
  environment variable. If unspecified, the Databricks SDK is used to identify the token based on
  the current environment.
- `columns (Optional[List[str]])`: Extra column names to include in response, in addition to the
  document id and text columns specified by `docs_id_column_name` and `text_column_name`.
- `filters_json (Optional[str])`: A JSON string specifying additional query filters.
  Example filters: `{"id <": 5}` selects records that have an `id` column value
  less than 5, and `{"id >=": 5, "id <": 10}` selects records that have an `id`
  column value greater than or equal to 5 and less than 10.
- `k (int)`: The number of documents to retrieve.
- `docs_id_column_name (str)`: The name of the column in the Databricks Vector Search Index
  containing document IDs.
- `text_column_name (str)`: The name of the column in the Databricks Vector Search Index
  containing document text to retrieve.

### Methods

#### `def forward(self, query: Union[str, List[float]], query_type: str = "ANN", filters_json: Optional[str] = None) -> dspy.Prediction:`

Retrieve documents from a Databricks Mosaic AI Vector Search Index that are relevant to the
specified query.

**Parameters:**

- `query (Union[str, List[float]])`: The query text or numeric query vector
  for which to retrieve relevant documents.
- `query_type (str)`: The type of search query to perform against the
  Databricks Vector Search Index. Must be either 'ANN' (approximate nearest neighbor) or 'HYBRID'
  (hybrid search).
- `filters_json (Optional[str])`: A JSON string specifying additional query filters.
  Example filters: `{"id <": 5}` selects records that have an `id` column value
  less than 5, and `{"id >=": 5, "id <": 10}` selects records that have an `id`
  column value greater than or equal to 5 and less than 10. If specified, this
  parameter overrides the `filters_json` parameter passed to the constructor.

**Returns:**

- `dspy.Prediction`: A `dotdict` containing retrieved documents. The schema is
  `{'docs': List[str], 'doc_ids': List[Any], extra_columns: List[Dict[str, Any]]}`.
  The `docs` entry contains the retrieved document content.

### Quickstart

To retrieve documents using Databricks Mosaic AI Vector Search, you must [create a
Databricks Mosaic AI Vector Search Index](https://docs.databricks.com/en/generative-ai/create-query-vector-search.html)
first.

The following example code demonstrates how to set up a Databricks Mosaic AI
[Direct Access Vector Search Index](https://docs.databricks.com/en/generative-ai/create-query-vector-search.html#create-a-vector-search-index)
and use the `DatabricksRM` DSPy retriever module to query the index. The example requires
the `databricks-vectorsearch` Python library to be installed.

```python
from databricks.vector_search.client import VectorSearchClient

# Create a Databricks Vector Search Endpoint
client = VectorSearchClient()
client.create_endpoint(
    name="your_vector_search_endpoint_name",
    endpoint_type="STANDARD"
)

# Create a Databricks Direct Access Vector Search Index
index = client.create_direct_access_index(
    endpoint_name="your_vector_search_endpoint_name",
    index_name="your_index_name",
    primary_key="id",
    embedding_dimension=1024,
    embedding_vector_column="text_vector",
    schema={
      "id": "int",
      "field2": "str",
      "field3": "float",
      "text_vector": "array<float>"
    }
)

# Create a DatabricksRM retriever and retrieve the top-3 most relevant documents from the
# Databricks Direct Access Vector Search Index corresponding to an example query
retriever = DatabricksRM(
    databricks_index_name = "your_index_name",
    docs_id_column_name="id",
    text_column_name="field2",
    k=3
)
retrieved_results = DatabricksRM(query="Example query text", query_type="hybrid"))
```


## End of deep-dive/retrieval_models_clients/DatabricksRM.md

---


## File: deep-dive/retrieval_models_clients/QdrantRM.md

# QdrantRM

[Qdrant](https://qdrant.tech/) is an open-source, high-performance vector search engine/database written in Rust. It can be used to retrieve semantically relevant passages to pass as context to your language model.

You can refer to the [end-to-end example](https://github.com/stanfordnlp/dspy/blob/main/examples/integrations/qdrant/qdrant_retriever_example.ipynb) demonstrating the use of DSPy and Qdrant.

## Setting Up QdrantRM

`QdrantRM` can be instantiated with any custom vectorizer and configured to return any payload field.

- `qdrant_collection_name (str)`: The name of the Qdrant collection.
- `qdrant_client (QdrantClient)`: An instance of `qdrant_client.QdrantClient`.
- `k (int, optional)`: The default number of top passages to retrieve. Default: 3.
- `document_field (str, optional)`: The key in the Qdrant payload with the content. Default: `"document"`.
- `vectorizer (BaseSentenceVectorizer, optional)`: An implementation `sentence_vectorizer.BaseSentenceVectorizer`. Default: `sentence_vectorizer.FastEmbedVectorizer`.
- `vector_name (str, optional)`: Name of the vector in the collection. Default: The first available vector name.

## Under the Hood

### `forward(self, query_or_queries: Union[str, list[str]], k: Optional[int] = None, filter: Optional[models.Filter] = None) -> dspy.Prediction`

**Parameters:**

- `query_or_queries (Union[str, List[str]])`: The query or queries to search for.
- `k (Optional[int])`: The number of top passages to retrieve. Defaults to `self.k`.
- `filter (Optional[qdrant_client.models.Filter])`: "Only include points satisfying the [filter conditions](https://qdrant.tech/documentation/concepts/filtering/)". Default: `None`.

**Returns:**

- `dspy.Prediction`: Contains the retrieved passages, each represented as a `dotdict` with a `long_text` attribute.

## Example Usage

```python
import os

from qdrant_client import QdrantClient

import dspy
from dsp.modules.sentence_vectorizer import OpenAIVectorizer
from dspy.retrieve.qdrant_rm import QdrantRM

os.environ["OPENAI_API_KEY"] = "<YOUR_OPENAI_API_KEY>"

client = QdrantClient(url="http://localhost:6333/")
vectorizer = OpenAIVectorizer(model="text-embedding-3-small")

qdrant_retriever = QdrantRM(
    qdrant_client=client,
    qdrant_collection_name="{collection_name}",
    vectorizer=vectorizer,
    document_field="text",
)

dspy.settings.configure(rm=qdrant_retriever)
retrieve = dspy.Retrieve()

retrieve("Some computer programs.")
```


## End of deep-dive/retrieval_models_clients/QdrantRM.md

---


## File: deep-dive/retrieval_models_clients/LancedbRM.md

# LancedbRM

[LanceDB](http://lancedb.com/) is a developer-friendly, open source database for AI. From hyper scalable vector search and advanced retrieval for RAG, to streaming training data and interactive exploration of large scale AI datasets, LanceDB is the best foundation for your AI application.

## Setting Up LancedbRM

`LancedbRM` can be instantiated with any custom vectorizer and configured to return any payload field.

- `table_name (str)`: The name of the table to query against.
- `persist_directory (str)`: directory where database is stored.
- `k (int, optional)`: The number of top passages to retrieve. Defaults to 3.

## Under the Hood

### `forward(self, query_or_queries: Union[str, list[str]], k: Optional[int] = None) -> dspy.Prediction`

**Parameters:**

- `query_or_queries (Union[str, List[str]])`: The query or queries to search for.
- `k (Optional[int])`: The number of top passages to retrieve. Defaults to `self.k`.

**Returns:**

- `dspy.Prediction`: Contains the retrieved passages, each represented as a `dotdict` with a `long_text` attribute.

## Example Usage

```python
import os
import pandas as pd

import dspy
from dspy.retrieve.lancedb import LancedbRM

from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import get_registry

from datasets import load_dataset

# load_dataset
ds = load_dataset("fancyzhx/dbpedia_14")
df = pd.DataFrame(ds['train'])

os.environ["OPENAI_API_KEY"] = "<YOUR_OPENAI_API_KEY>"

uri = 'tmp/db'
table_name = 'passages'
db = lancedb.connect(uri)
model = get_registry().get("sentence-transformers").create(name="BAAI/bge-small-en-v1.5", device="cpu")

class Passages(LanceModel):
    text: str = model.SourceField()
    vector: Vector(model.ndims()) = model.VectorField()

table = db.create_table(table_name, schema=Passages)
table.add(df['content'])


lancedb_retriever = LancedbRM(
    table_name=table_name,
    persist_directory=uri,
)

dspy.settings.configure(rm=lancedb_retriever)
retrieve = dspy.Retrieve()

retrieve("Integrated Circuits")
```


## End of deep-dive/retrieval_models_clients/LancedbRM.md

---


## File: deep-dive/retrieval_models_clients/SnowflakeRM.md

# SnowflakeRM

### Constructor

Initialize an instance of the `SnowflakeRM` class, which enables user to leverage the Cortex Search service for hybrid retrieval. Before using this, ensure the Cortex Search service is configured as outlined in the documentation [here](https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-search/cortex-search-overview#overview)

```python
SnowflakeRM(
     snowflake_session: object,
     cortex_search_service: str,
     snowflake_database: str,
     snowflake_schema: dict,
     auto_filter:bool,
     k: int = 3,
)
```

**Parameters:**

- `snowflake_session (str)`: Snowflake Snowpark session for connecting to Snowflake.
- `cortex_search_service (str)`: The name of the Cortex Search service to be used.
- `snowflake_database (str)`: The name of the Snowflake database to be used with the Cortex Search service.
- `snowflake_schema (str)`: The name of the Snowflake schema to be used with the Cortex Search service.
- `auto_filter (bool)`: Auto-generate metadata filter and push it down to Cortex Search service prior to retrieving results.
- `k (int, optional)`: The number of top passages to retrieve. Defaults to 3.

### Methods

#### `def forward(self,query_or_queries: Union[str, list[str]],response_columns:list[str],filters:dict = None, k: Optional[int] = None)-> dspy.Prediction:`

Query the Cortex Search service to retrieve the top k relevant results given a query.

**Parameters:**

- `query_or_queries` (_Union[str, List[str]]_): The query or list of queries to search for.
- `retrieval_columns` (str)`: A list of columns to return for each relevant result in the response.
- `search_filter` (_Optional[dict]_): Optional filter object used for filtering results based on data in the ATTRIBUTES columns. See [Filter syntax](https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-search/query-cortex-search-service#filter-syntax)
- `k` (_Optional[int]_): The number of results to retrieve. If not specified, defaults to the value set during initialization.

**Returns:**

- `dspy.Prediction`: Contains the retrieved passages, each represented as a `dotdict` with schema `[{"long_text": str}]`

### Quickstart

To support passage retrieval from a Snowflake table with this integration, a [Cortex Search](https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-search/cortex-search-overview) endpoint must first be configured.

```python
from dspy.retrieve.snowflake_rm import SnowflakeRM
from snowflake.snowpark import Session
import os

connection_parameters = {

    "account": os.getenv('SNOWFLAKE_ACCOUNT'),
    "user": os.getenv('SNOWFLAKE_USER'),
    "password": os.getenv('SNOWFLAKE_PASSWORD'),
    "role": os.getenv('SNOWFLAKE_ROLE'),
    "warehouse": os.getenv('SNOWFLAKE_WAREHOUSE'),
    "database": os.getenv('SNOWFLAKE_DATABASE'),
    "schema": os.getenv('SNOWFLAKE_SCHEMA')}

# Establish connection to Snowflake
snowpark = Session.builder.configs(connection_parameters).create()

snowflake_retriever = SnowflakeRM(snowflake_session=snowpark,
    cortex_search_service="<YOUR_CORTEX_SEARCH_SERVICE_NAME>",
    snowflake_database="<YOUR_SNOWFLAKE_DATABASE_NAME>",
    snowflake_schema="<YOUR_SNOWFLAKE_SCHEMA_NAME>",
    auto_filter=True,
    k = 5)

results = snowflake_retriever("Explore the meaning of life",
    response_columns=["<NAME_OF_INDEXED_COLUMN>","<NAME_OF_ATTRIBUTE_COLUMN"])

for result in results:
    print("Document:", result.long_text, "\n")
```


## End of deep-dive/retrieval_models_clients/SnowflakeRM.md

---


## File: deep-dive/retrieval_models_clients/Azure.md

import AuthorDetails from '@site/src/components/AuthorDetails';

# AzureAISearch

A retrieval module that utilizes Azure AI Search to retrieve top passages for a given query.

## Prerequisites

```bash
pip install azure-search-documents
```

## Setting up the AzureAISearchRM Client

The constructor initializes an instance of the `AzureAISearchRM` class and sets up parameters for sending queries and retrieving results with the Azure AI Search server.

- `search_service_name` (str): The name of the Azure AI Search service.
- `search_api_key` (str): The API key for accessing the Azure AI Search service.
- `search_index_name` (str): The name of the search index in the Azure AI Search service.
- `field_text` (str): The name of the field containing text content in the search index. This field will be mapped to the "content" field in the dsp framework.
- `field_vector` (Optional[str]): The name of the field containing vector content in the search index.
- `k` (int, optional): The default number of top passages to retrieve. Defaults to 3.
- `azure_openai_client` (Optional[openai.AzureOpenAI]): An instance of the AzureOpenAI client. Either openai_client or embedding_func must be provided. Defaults to None.
- `openai_embed_model` (Optional[str]): The name of the OpenAI embedding model. Defaults to "text-embedding-ada-002".
- `embedding_func` (Optional[Callable]): A function for generating embeddings. Either openai_client or embedding_func must be provided. Defaults to None.
- `semantic_ranker` (bool, optional): Whether to use semantic ranking. Defaults to False.
- `filter` (str, optional): Additional filter query. Defaults to None.
- `query_language` (str, optional): The language of the query. Defaults to "en-Us".
- `query_speller` (str, optional): The speller mode. Defaults to "lexicon".
- `use_semantic_captions` (bool, optional): Whether to use semantic captions. Defaults to False.
- `query_type` (Optional[QueryType], optional): The type of query. Defaults to QueryType.FULL.
- `semantic_configuration_name` (str, optional): The name of the semantic configuration. Defaults to None.
- `is_vector_search` (Optional[bool]): Whether to enable vector search. Defaults to False.
- `is_hybrid_search` (Optional[bool]): Whether to enable hybrid search. Defaults to False.
- `is_fulltext_search` (Optional[bool]): Whether to enable fulltext search. Defaults to True.
- `vector_filter_mode` (Optional[VectorFilterMode]): The vector filter mode. Defaults to None.


**Available Query Types:**

- SIMPLE
    """Uses the simple query syntax for searches. Search text is interpreted using a simple query
    #: language that allows for symbols such as +, * and "". Queries are evaluated across all
    #: searchable fields by default, unless the searchFields parameter is specified."""
- FULL
    """Uses the full Lucene query syntax for searches. Search text is interpreted using the Lucene
    #: query language which allows field-specific and weighted searches, as well as other advanced
    #: features."""
- SEMANTIC
    """Best suited for queries expressed in natural language as opposed to keywords. Improves
    #: precision of search results by re-ranking the top search results using a ranking model trained
    #: on the Web corpus.""

    More Details: https://learn.microsoft.com/en-us/azure/search/search-query-overview

**Available Vector Filter Mode:**

- POST_FILTER = "postFilter"
    """The filter will be applied after the candidate set of vector results is returned. Depending on
    #: the filter selectivity, this can result in fewer results than requested by the parameter 'k'."""

- PRE_FILTER = "preFilter"
    """The filter will be applied before the search query."""

    More Details: https://learn.microsoft.com/en-us/azure/search/vector-search-filters

**Note**

- The `AzureAISearchRM` client allows you to perform Vector search, Hybrid search, or Full text search.
- By default, the `AzureAISearchRM` client uses the Azure OpenAI Client for generating embeddings. If you want to use something else, you can provide your custom embedding_func, but either the openai_client or embedding_func must be provided.
- If you need to enable semantic search, either with vector, hybrid, or full text search, then set the `semantic_ranker` flag to True.
- If `semantic_ranker` is True, always set the `query_type` to QueryType.SEMANTIC and always provide the `semantic_configuration_name`.

Example of the AzureAISearchRM constructor:

```python
AzureAISearchRM(
    search_service_name: str,
    search_api_key: str,
    search_index_name: str,
    field_text: str,
    field_vector: Optional[str] = None,
    k: int = 3,
    azure_openai_client: Optional[openai.AzureOpenAI] = None,
    openai_embed_model: Optional[str] = "text-embedding-ada-002",
    embedding_func: Optional[Callable] = None,
    semantic_ranker: bool = False,
    filter: str = None,
    query_language: str = "en-Us",
    query_speller: str = "lexicon",
    use_semantic_captions: bool = False,
    query_type: Optional[QueryType] = QueryType.FULL,
    semantic_configuration_name: str = None,
    is_vector_search: Optional[bool] = False,
    is_hybrid_search: Optional[bool] = False,
    is_fulltext_search: Optional[bool] = True,
    vector_filter_mode: Optional[VectorFilterMode.PRE_FILTER] = None
)
```

## Under the Hood

### `forward(self, query_or_queries: Union[str, List[str]], k: Optional[int] = None) -> dspy.Prediction`

**Parameters:**

- `query_or_queries` (Union[str, List[str]]): The query or queries to search for.
- `k` (_Optional[int]_, _optional_): The number of results to retrieve. If not specified, defaults to the value set during initialization.

**Returns:**

- `dspy.Prediction`: Contains the retrieved passages, each represented as a `dotdict` with a `long_text` attribute.

Internally, the method handles the specifics of preparing the request query to the Azure AI Search service and corresponding payload to obtain the response.

The function handles the retrieval of the top-k passages based on the provided query.

## Sending Retrieval Requests via AzureAISearchRM Client

1. _**Recommended**_ Configure default RM using `dspy.configure`.

This allows you to define programs in DSPy and have DSPy internally conduct retrieval using `dsp.retrieve` on the query on the configured RM.

```python
import dspy
from dspy.retrieve.azureaisearch_rm import AzureAISearchRM

azure_search = AzureAISearchRM(
    "search_service_name",
    "search_api_key",
    "search_index_name",
    "field_text",
    "k"=3
)

dspy.settings.configure(rm=azure_search)
retrieve = dspy.Retrieve(k=3)
retrieval_response = retrieve("What is Thermodynamics").passages

for result in retrieval_response:
    print("Text:", result, "\n")
```

2. Generate responses using the client directly.

```python
from dspy.retrieve.azureaisearch_rm import AzureAISearchRM

azure_search = AzureAISearchRM(
    "search_service_name",
    "search_api_key",
    "search_index_name",
    "field_text",
    "k"=3
)

retrieval_response = azure_search("What is Thermodynamics", k=3)
for result in retrieval_response:
    print("Text:", result.long_text, "\n")
```

3. Example of Semantic Hybrid Search.

```python
from dspy.retrieve.azureaisearch_rm import AzureAISearchRM

azure_search = AzureAISearchRM(
    search_service_name="search_service_name",
    search_api_key="search_api_key",
    search_index_name="search_index_name",
    field_text="field_text",
    field_vector="field_vector",
    k=3,
    azure_openai_client="azure_openai_client",
    openai_embed_model="text-embedding-ada-002"
    semantic_ranker=True,
    query_type=QueryType.SEMANTIC,
    semantic_configuration_name="semantic_configuration_name",
    is_hybrid_search=True,
)

retrieval_response = azure_search("What is Thermodynamics", k=3)
for result in retrieval_response:
    print("Text:", result.long_text, "\n")
```

***

<AuthorDetails name="Prajapati Harishkumar Kishorkumar"/>

## End of deep-dive/retrieval_models_clients/Azure.md

---


## File: deep-dive/data-handling/loading-custom-data.md

---
sidebar_position: 3
---

!!! warning "This page is outdated and may not be fully accurate in DSPy 2.5"

# Creating a Custom Dataset

We've seen how to work with with `Example` objects and use the `HotPotQA` class to load the HuggingFace HotPotQA dataset as a list of `Example` objects. But in production, such structured datasets are rare. Instead, you'll find yourself working on a custom dataset and might question: how do I create my own dataset or what format should it be?

In DSPy, your dataset is a list of `Examples`, which we can accomplish in two ways:

* **Recommended: The Pythonic Way:** Using native python utility and logic.
* **Advanced: Using DSPy's `Dataset` class**

## Recommended: The Pythonic Way

To create a list of `Example` objects, we can simply load data from the source and formulate it into a Python list. Let's load an example CSV `sample.csv` that contains 3 fields: (**context**, **question** and **summary**) via Pandas. From there, we can construct our data list.

```python
import pandas as pd

df = pd.read_csv("sample.csv")
print(df.shape)
```
**Output:**
```text
(1000, 3)
```
```python
dataset = []

for context, question, answer in df.values:
    dataset.append(dspy.Example(context=context, question=question, answer=answer).with_inputs("context", "question"))

print(dataset[:3])
```
**Output:**
```python
[Example({'context': nan, 'question': 'Which is a species of fish? Tope or Rope', 'answer': 'Tope'}) (input_keys={'question', 'context'}),
 Example({'context': nan, 'question': 'Why can camels survive for long without water?', 'answer': 'Camels use the fat in their humps to keep them filled with energy and hydration for long periods of time.'}) (input_keys={'question', 'context'}),
 Example({'context': nan, 'question': "Alice's parents have three daughters: Amy, Jessy, and what’s the name of the third daughter?", 'answer': 'The name of the third daughter is Alice'}) (input_keys={'question', 'context'})]
```

While this is fairly simple, let's take a look at how loading datasets would look in DSPy - via the DSPythonic way!

## Advanced: Using DSPy's `Dataset` class (Optional)

Let's take advantage of the `Dataset` class we defined in the previous article to accomplish the preprocessing: 

* Load data from CSV to a dataframe.
* Split the data to train, dev and test splits.
* Populate `_train`, `_dev` and `_test` class attributes. Note that these attributes should be a list of dictionary, or an iterator over mapping like HuggingFace Dataset, to make it work.

This is all done through the `__init__` method, which is the only method we have to implement.

```python
import pandas as pd
from dspy.datasets.dataset import Dataset

class CSVDataset(Dataset):
    def __init__(self, file_path, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        df = pd.read_csv(file_path)
        self._train = df.iloc[0:700].to_dict(orient='records')

        self._dev = df.iloc[700:].to_dict(orient='records')

dataset = CSVDataset("sample.csv")
print(dataset.train[:3])
```
**Output:**
```text
[Example({'context': nan, 'question': 'Which is a species of fish? Tope or Rope', 'answer': 'Tope'}) (input_keys={'question', 'context'}),
 Example({'context': nan, 'question': 'Why can camels survive for long without water?', 'answer': 'Camels use the fat in their humps to keep them filled with energy and hydration for long periods of time.'}) (input_keys={'question', 'context'}),
 Example({'context': nan, 'question': "Alice's parents have three daughters: Amy, Jessy, and what’s the name of the third daughter?", 'answer': 'The name of the third daughter is Alice'}) (input_keys={'question', 'context'})]
```

Let's understand the code step by step:

* It inherits the base `Dataset` class from DSPy. This inherits all the useful data loading/processing functionality.
* We load the data in CSV into a DataFrame.
* We get the **train** split i.e first 700 rows in the DataFrame and convert it to lists of dicts using `to_dict(orient='records')` method and is then assigned to `self._train`.
* We get the **dev** split i.e first 300 rows in the DataFrame and convert it to lists of dicts using `to_dict(orient='records')` method and is then assigned to `self._dev`.

Using the Dataset base class now makes loading custom datasets incredibly easy and avoids having to write all that boilerplate code ourselves for every new dataset.

!!! caution

    We did not populate `_test` attribute in the above code, which is fine and won't cause any unnecessary error as such. However it'll give you an error if you try to access the test split.

    ```python
    dataset.test[:5]
    ```
    ****
    ```text
    ---------------------------------------------------------------------------
    AttributeError                            Traceback (most recent call last)
    <ipython-input-59-5202f6de3c7b> in <cell line: 1>()
    ----> 1 dataset.test[:5]

    /usr/local/lib/python3.10/dist-packages/dspy/datasets/dataset.py in test(self)
        51     def test(self):
        52         if not hasattr(self, '_test_'):
    ---> 53             self._test_ = self._shuffle_and_sample('test', self._test, self.test_size, self.test_seed)
        54 
        55         return self._test_

    AttributeError: 'CSVDataset' object has no attribute '_test'
    ```

    To prevent that you'll just need to make sure `_test` is not `None` and populated with the appropriate data.

You can override the methods in `Dataset` class to customize your class even more. 

In summary, the Dataset base class provides a simplistic way to load and preprocess custom datasets with minimal code!


## End of deep-dive/data-handling/loading-custom-data.md

---


## File: deep-dive/data-handling/examples.md

---
sidebar_position: 1
---

!!! warning "This page is outdated and may not be fully accurate in DSPy 2.5"

# Examples in DSPy

Working in DSPy involves training sets, development sets, and test sets. This is like traditional ML, but you usually need far fewer labels (or zero labels) to use DSPy effectively.

The core data type for data in DSPy is `Example`. You will use **Examples** to represent items in your training set and test set. 

DSPy **Examples** are similar to Python `dict`s but have a few useful utilities. Your DSPy modules will return values of the type `Prediction`, which is a special sub-class of `Example`.

## Creating an `Example`

When you use DSPy, you will do a lot of evaluation and optimization runs. Your individual datapoints will be of type `Example`:

```python
qa_pair = dspy.Example(question="This is a question?", answer="This is an answer.")

print(qa_pair)
print(qa_pair.question)
print(qa_pair.answer)
```
**Output:**
```text
Example({'question': 'This is a question?', 'answer': 'This is an answer.'}) (input_keys=None)
This is a question?
This is an answer.
```

Examples can have any field keys and any value types, though usually values are strings.

```text
object = Example(field1=value1, field2=value2, field3=value3, ...)
```

## Specifying Input Keys

In traditional ML, there are separated "inputs" and "labels".

In DSPy, the `Example` objects have a `with_inputs()` method, which can mark specific fields as inputs. (The rest are just metadata or labels.)

```python
# Single Input.
print(qa_pair.with_inputs("question"))

# Multiple Inputs; be careful about marking your labels as inputs unless you mean it.
print(qa_pair.with_inputs("question", "answer"))
```

This flexibility allows for customized tailoring of the `Example` object for different DSPy scenarios.

When you call `with_inputs()`, you get a new copy of the example. The original object is kept unchanged.


## Element Access and Updation

Values can be accessed using the `.`(dot) operator. You can access the value of key `name` in defined object `Example(name="John Doe", job="sleep")` through `object.name`. 

To access or exclude certain keys, use `inputs()` and `labels()` methods to return new Example objects containing only input or non-input keys, respectively.

```python
article_summary = dspy.Example(article= "This is an article.", summary= "This is a summary.").with_inputs("article")

input_key_only = article_summary.inputs()
non_input_key_only = article_summary.labels()

print("Example object with Input fields only:", input_key_only)
print("Example object with Non-Input fields only:", non_input_key_only)
```

**Output**
```
Example object with Input fields only: Example({'article': 'This is an article.'}) (input_keys=None)
Example object with Non-Input fields only: Example({'summary': 'This is a summary.'}) (input_keys=None)
```

To exclude keys, use `without()`:

```python
article_summary = dspy.Example(context="This is an article.", question="This is a question?", answer="This is an answer.", rationale= "This is a rationale.").with_inputs("context", "question")

print("Example object without answer & rationale keys:", article_summary.without("answer", "rationale"))
```

**Output**
```
Example object without answer & rationale keys: Example({'context': 'This is an article.', 'question': 'This is a question?'}) (input_keys=None)
```

Updating values is simply assigning a new value using the `.` operator.

```python
article_summary.context = "new context"
```

## Iterating over Example

Iteration in the `Example` class also functions like a dictionary, supporting methods like `keys()`, `values()`, etc: 

```python
for k, v in article_summary.items():
    print(f"{k} = {v}")
```

**Output**

```text
context = This is an article.
question = This is a question?
answer = This is an answer.
rationale = This is a rationale.
```


## End of deep-dive/data-handling/examples.md

---


## File: deep-dive/data-handling/built-in-datasets.md

---
sidebar_position: 2
---

!!! warning "This page is outdated and may not be fully accurate in DSPy 2.5"

# Utilizing Built-in Datasets

It's easy to use your own data in DSPy: a dataset is just a list of `Example` objects. Using DSPy well involves being able to find and re-purpose existing datasets for your own pipelines in new ways; DSPy makes this a particularly powerful strategy.

For convenience, DSPy currently also provides support for the following dataset out of the box:

* **HotPotQA** (multi-hop question answering)
* **GSM8k** (math questions)
* **Color** (basic dataset of colors)


## Loading HotPotQA

HotPotQA is which is a collection of question-answer pairs.

```python
from dspy.datasets import HotPotQA

dataset = HotPotQA(train_seed=1, train_size=5, eval_seed=2023, dev_size=50, test_size=0)

print(dataset.train)
```
**Output:**
```text
[Example({'question': 'At My Window was released by which American singer-songwriter?', 'answer': 'John Townes Van Zandt'}) (input_keys=None),
 Example({'question': 'which  American actor was Candace Kita  guest starred with ', 'answer': 'Bill Murray'}) (input_keys=None),
 Example({'question': 'Which of these publications was most recently published, Who Put the Bomp or Self?', 'answer': 'Self'}) (input_keys=None),
 Example({'question': 'The Victorians - Their Story In Pictures is a documentary series written by an author born in what year?', 'answer': '1950'}) (input_keys=None),
 Example({'question': 'Which magazine has published articles by Scott Shaw, Tae Kwon Do Times or Southwest Art?', 'answer': 'Tae Kwon Do Times'}) (input_keys=None)]
```

We just loaded trainset (5 examples) and devset (50 examples). Each example in our training set contains just a question and its (human-annotated) answer. As you can see, it is loaded as a list of `Example` objects. However, one thing to note is that it doesn't set the input keys implicitly, so that is something that we'll need to do!!

```python
trainset = [x.with_inputs('question') for x in dataset.train]
devset = [x.with_inputs('question') for x in dataset.dev]

print(trainset)
```
**Output:**
```text
[Example({'question': 'At My Window was released by which American singer-songwriter?', 'answer': 'John Townes Van Zandt'}) (input_keys={'question'}),
 Example({'question': 'which  American actor was Candace Kita  guest starred with ', 'answer': 'Bill Murray'}) (input_keys={'question'}),
 Example({'question': 'Which of these publications was most recently published, Who Put the Bomp or Self?', 'answer': 'Self'}) (input_keys={'question'}),
 Example({'question': 'The Victorians - Their Story In Pictures is a documentary series written by an author born in what year?', 'answer': '1950'}) (input_keys={'question'}),
 Example({'question': 'Which magazine has published articles by Scott Shaw, Tae Kwon Do Times or Southwest Art?', 'answer': 'Tae Kwon Do Times'}) (input_keys={'question'})]
```

DSPy typically requires very minimal labeling. Whereas your pipeline may involve six or seven complex steps, you only need labels for the initial question and the final answer. DSPy will bootstrap any intermediate labels needed to support your pipeline. If you change your pipeline in any way, the data bootstrapped will change accordingly!


## Advanced: Inside DSPy's `Dataset` class (Optional)

We've seen how you can use `HotPotQA` dataset class and load the HotPotQA dataset, but how does it actually work? The `HotPotQA` class inherits from the `Dataset` class, which takes care of the conversion of the data loaded from a source into train-test-dev split, all of which are *list of examples*. In the `HotPotQA` class, you only implement the `__init__` method, where you populate the splits from HuggingFace into the variables `_train`, `_test` and `_dev`. The rest of the process is handled by methods in the `Dataset` class.

![Dataset Loading Process in HotPotQA Class](./img/data-loading.png)

But how do the methods of the `Dataset` class convert the data from HuggingFace? Let's take a deep breath and think step by step...pun intended. In example above, we can see the splits accessed by `.train`, `.dev` and `.test` methods, so let's take a look at the implementation of the `train()` method:

```python
@property
def train(self):
    if not hasattr(self, '_train_'):
        self._train_ = self._shuffle_and_sample('train', self._train, self.train_size, self.train_seed)

    return self._train_
```

As you can see, the `train()` method serves as a property, not a regular method. Within this property, it first checks if the `_train_` attribute exists. If not, it calls the `_shuffle_and_sample()` method to process the `self._train` where the HuggingFace dataset is loaded. Let's see the  `_shuffle_and_sample()` method:

```python
def _shuffle_and_sample(self, split, data, size, seed=0):
    data = list(data)
    base_rng = random.Random(seed)

    if self.do_shuffle:
        base_rng.shuffle(data)

    data = data[:size]
    output = []

    for example in data:
        output.append(Example(**example, dspy_uuid=str(uuid.uuid4()), dspy_split=split))
    
        return output
```

The `_shuffle_and_sample()` method does two things:

* It shuffles the data if `self.do_shuffle` is True.
* It then takes a sample of size `size` from the shuffled data.
* It then loops through the sampled data and converts each element in `data` into an `Example` object. The `Example` along with example data also contains a unique ID, and the split name.

Converting the raw examples into `Example` objects allows the Dataset class to process them in a standardized way later. For example, the collate method, which is used by the PyTorch DataLoader, expects each item to be an `Example`.

To summarize, the `Dataset` class handles all the necessary data processing and provides a simple API to access the different splits. This differentiates from the dataset classes like HotpotQA which require only definitions on how to load the raw data.


## End of deep-dive/data-handling/built-in-datasets.md

---


## File: deep-dive/language_model_clients/lm_remote_models.md



## End of deep-dive/language_model_clients/lm_remote_models.md

---


## File: deep-dive/language_model_clients/lm_local_models/HFClientTGI.md

import AuthorDetails from '@site/src/components/AuthorDetails';

# HFClientTGI

!!! warning "This page is outdated and may not be fully accurate in DSPy 2.5"


## Prerequisites

Docker must be installed on your system. If you don't have Docker installed, you can get it from [here](https://docs.docker.com/get-docker/).

1. Clone the Text-Generation-Inference repository from GitHub by executing the following command:

   ```
   git clone https://github.com/huggingface/text-generation-inference.git
   ```

2. Change into the cloned repository directory:

   ```
   cd text-generation-inference
   ```

3. Execute the Docker command under the "Get Started" section to run the server:


   ```
   model=meta-llama/Llama-2-7b-hf # set to the specific Hugging Face model ID you wish to use.
   num_shard=2 # set to the number of shards you wish to use.
   volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

   docker run --gpus all --shm-size 1g -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:0.9 --model-id $model --num-shard $num_shard
   ```

   This command will start the server and make it accessible at `http://localhost:8080`.

If you want to connect to [Meta Llama 2 models](https://huggingface.co/meta-llama), make sure to use version 9.3 (or higher) of the docker image (ghcr.io/huggingface/text-generation-inference:0.9.3) and pass in your huggingface token as an environment variable.

```
   docker run --gpus all --shm-size 1g -p 8080:80 -v $volume:/data -e HUGGING_FACE_HUB_TOKEN={your_token} ghcr.io/huggingface/text-generation-inference:0.9.3 --model-id $model --num-shard $num_shard
```

## Using the TGI Client

After setting up the text-generation-inference server and ensuring that it displays "Connected" when it's running, you can interact with it using the `HFClientTGI`.

Initialize the `HFClientTGI` within your program with the desired parameters. Here is an example call:

```python
tgi_llama2 = dspy.HFClientTGI(model="meta-llama/Llama-2-7b-hf", port=8080, url="http://localhost")
```

Customize the `model`, `port`, and `url` according to your requirements. The `model` parameter should be set to the specific Hugging Face model ID you wish to use. 

## Sending Requests via TGI Client

1) _**Recommended**_ Configure default LM using `dspy.configure`.

This allows you to define programs in DSPy and simply call modules on your input fields, having DSPy internally call the prompt on the configured LM.

```python
dspy.configure(lm=tgi_llama2)

#Example DSPy CoT QA program
qa = dspy.ChainOfThought('question -> answer')

response = qa(question="What is the capital of Paris?") #Prompted to tgi_llama2
print(response.answer)
```

2) Generate responses using the client directly.

```python
response = tgi_llama2._generate(prompt='What is the capital of Paris?')
print(response)
```

## Under the Hood

### `__init__(self, model, port, url="http://future-hgx-1", http_request_kwargs=None, **kwargs)`

The constructor initializes the `HFModel` base class to support the handling of prompting HuggingFace models. It configures the client for communicating with the hosted TGI server to generate requests. This requires the following parameters:

- `model` (_str_): ID of Hugging Face model connected to the TGI server.
- `port` (_int_ or _list_): Port for communicating to the TGI server. This can be a single port number (`8080`) or a list of TGI ports (`[8080, 8081, 8082]`) to route the requests to.
- `url` (_str_): Base URL of hosted TGI server. This will often be `"http://localhost"`.
- `http_request_kwargs` (_dict_): Dictionary of additional keyword arguments to pass to the HTTP request function to the TGI server. This is `None` by default. 
- `**kwargs`: Additional keyword arguments to configure the TGI client.

Example of the TGI constructor:

```python
class HFClientTGI(HFModel):
    def __init__(self, model, port, url="http://future-hgx-1", http_request_kwargs=None, **kwargs):
```

### `_generate(self, prompt, **kwargs) -> dict`

**Parameters:**
- `prompt` (_str_): Prompt to send to model hosted on TGI server.
- `**kwargs`: Additional keyword arguments for completion request.

**Returns:**
- `dict`: dictionary with `prompt` and list of response `choices`.

Internally, the method handles the specifics of preparing the request prompt and corresponding payload to obtain the response. 

After generation, the method parses the JSON response received from the server and retrieves the output through `json_response["generated_text"]`. This is then stored in the `completions` list.

If the JSON response includes the additional `details` argument and correspondingly, the `best_of_sequences` within `details`, this indicates multiple sequences were generated. This is also usually the case when `best_of > 1` in the initialized kwargs. Each of these sequences is accessed through `x["generated_text"]` and added to the `completions` list.

Lastly, the method constructs the response dictionary with two keys: the original request `prompt` and `choices`, a list of dictionaries representing generated completions with the key `text` holding the response's generated text.

## FAQs

1. If your model doesn't require any shards, you still need to set a value for `num_shard`, but you don't need to include the parameter `--num-shard` on the command line.

2. If your model runs into any "token exceeded" issues, you can set the following parameters on the command line to adjust the input length and token limit:
   - `--max-input-length`: Set the maximum allowed input length for the text.
   - `--max-total-tokens`: Set the maximum total tokens allowed for text generation.

Please refer to the [official Text-Generation-Inference repository](https://github.com/huggingface/text-generation-inference) for more detailed information and documentation.


***

<AuthorDetails name="Arnav Singhvi"/>

## End of deep-dive/language_model_clients/lm_local_models/HFClientTGI.md

---


## File: deep-dive/language_model_clients/lm_local_models/Ollama.md

# OllamaLocal

!!! warning "This page is outdated and may not be fully accurate in DSPy 2.5"

!!! note
    Adapted from documentation provided by https://github.com/insop

Ollama is a good software tool that allows you to run LLMs locally, such as Mistral, Llama2, and Phi.
The following are the instructions to install and run Ollama.

### Prerequisites

Install Ollama by following the instructions from this page:

- https://ollama.ai

Download model: `ollama pull`

Download a model by running the `ollama pull` command. You can download Mistral, Llama2, and Phi.

```bash
# download mistral
ollama pull mistral
```

Here is the list of other models you can download:
- https://ollama.ai/library

### Running Ollama model

Run model: `ollama run`

You need to start the model server with the `ollama run` command.

```bash
# run mistral
ollama run mistral
```

### Sending requests to the server

Here is the code to load a model through Ollama:

```python
lm = dspy.OllamaLocal(model='mistral')
```


## End of deep-dive/language_model_clients/lm_local_models/Ollama.md

---


## File: deep-dive/language_model_clients/lm_local_models/LlamaCpp.md

# LlamaCpp

!!! warning "This page is outdated and may not be fully accurate in DSPy 2.5"


## Prerequisites

Install Llama Cpp Python by following the instructions provided in the [Llama Cpp Python repository](https://github.com/abetlen/llama-cpp-python).

```shell
pip install llama-cpp-python
```

alternatively, to install with CUDA support:

```shell
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python
```


### Initializing the Llama model

Initialize the model within your program with the desired parameters.

```python
from llama_cpp import Llama

llm = Llama(
      model_path="./sppo_finetuned_llama_3_8b.gguf",
      n_gpu_layers=-1,
      n_ctx=0,
      verbose=False
)
```


### Sending requests to the model

After initializing the Llama model, you can interact with it using the `LlamaCpp` client.

```python
import dspy

llamalm = dspy.LlamaCpp(model="llama", llama_model=llm,  model_type="chat", temperature=0.4)
dspy.settings.configure(lm=llamalm)


#Define a simple signature for basic question answering
class BasicQA(dspy.Signature):
    """Answer questions with short factoid answers."""
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")

#Pass signature to Predict module
generate_answer = dspy.Predict(BasicQA)

# Call the predictor on a particular input.
question='What is the color of the sky?'
pred = generate_answer(question=question)

print(f"Question: {question}")
print(f"Predicted Answer: {pred.answer}")


```

## End of deep-dive/language_model_clients/lm_local_models/LlamaCpp.md

---


## File: deep-dive/language_model_clients/lm_local_models/TensorRTLLM.md

# TensorRTModel

!!! warning "This page is outdated and may not be fully accurate in DSPy 2.5"

TensorRT LLM by Nvidia happens to be one of the most optimized inference engines to run open-source Large Language Models locally or in production.

### Prerequisites

Install TensorRT LLM by the following instructions [here](https://nvidia.github.io/TensorRT-LLM/installation/linux.html). You need to install `dspy` inside the same Docker environment in which `tensorrt` is installed.

In order to use this module, you should have the model weights file in engine format. To understand how we convert weights in torch (from HuggingFace models) to TensorRT engine format, you can check out [this documentation](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/llama#build-tensorrt-engines).

### Running TensorRT model inside dspy

```python
from dspy import TensorRTModel

engine_dir = "<your-path-to-engine-dir>"
model_name_or_path = "<hf-model-id-or-path-to-tokenizer>"

model = TensorRTModel(engine_dir=engine_dir, model_name_or_path=model_name_or_path)
```

You can perform more customization on model loading based on the following example. Below is a list of optional parameters that are supported while initializing the `dspy` TensorRT model.

- **use_py_session** (`bool`, optional): Whether to use a Python session or not. Defaults to `False`.
- **lora_dir** (`str`): The directory of LoRA adapter weights.
- **lora_task_uids** (`List[str]`): List of LoRA task UIDs; use `-1` to disable the LoRA module.
- **lora_ckpt_source** (`str`): The source of the LoRA checkpoint.

If `use_py_session` is set to `False`, the following kwargs are supported (This runs in C++ runtime):

- **max_batch_size** (`int`, optional): The maximum batch size. Defaults to `1`.
- **max_input_len** (`int`, optional): The maximum input context length. Defaults to `1024`.
- **max_output_len** (`int`, optional): The maximum output context length. Defaults to `1024`.
- **max_beam_width** (`int`, optional): The maximum beam width, similar to `n` in OpenAI API. Defaults to `1`.
- **max_attention_window_size** (`int`, optional): The attention window size that controls the sliding window attention / cyclic KV cache behavior. Defaults to `None`.
- **sink_token_length** (`int`, optional): The sink token length. Defaults to `1`.

> Please note that you need to complete the build processes properly before applying these customizations, because a lot of customization depends on how the model engine was built. You can learn more [here](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/llama#build-tensorrt-engines).

Now to run the model, we need to add the following code:

```python
response = model("hello")
```

This gives this result:

```
["nobody is perfect, and we all have our own unique struggles and challenges. But what sets us apart is how we respond to those challenges. Do we let them define us, or do we use them as opportunities to grow and learn?\nI know that I have my own personal struggles, and I'm sure you do too. But I also know that we are capable of overcoming them, and becoming the best versions of ourselves. So let's embrace our imperfections, and use them to fuel our growth and success.\nRemember, nobody is perfect, but everybody has the potential to be amazing. So let's go out there and make it happen!"]
```

You can also invoke chat mode by just changing the prompt to chat format like this:

```python
prompt = [{"role":"user", "content":"hello"}]
response = model(prompt)

print(response)
```

Output:

```
[" Hello! It's nice to meet you. Is there something I can help you with or would you like to chat?"]
```

Here are some optional parameters that are supported while doing generation:

- **max_new_tokens** (`int`): The maximum number of tokens to output. Defaults to `1024`.
- **max_attention_window_size** (`int`): Defaults to `None`.
- **sink_token_length** (`int`): Defaults to `None`.
- **end_id** (`int`): The end of sequence ID of the tokenizer, defaults to the tokenizer's default end ID.
- **pad_id** (`int`): The pad sequence ID of the tokenizer, defaults to the tokenizer's default end ID.
- **temperature** (`float`): The temperature to control probabilistic behavior in generation. Defaults to `1.0`.
- **top_k** (`int`): Defaults to `1`.
- **top_p** (`float`): Defaults to `1`.
- **num_beams** (`int`): The number of responses to generate. Defaults to `1`.
- **length_penalty** (`float`): Defaults to `1.0`.
- **repetition_penalty** (`float`): Defaults to `1.0`.
- **presence_penalty** (`float`): Defaults to `0.0`.
- **frequency_penalty** (`float`): Defaults to `0.0`.
- **early_stopping** (`int`): Use this only when `num_beams` > 1. Defaults to `1`.


## End of deep-dive/language_model_clients/lm_local_models/TensorRTLLM.md

---


## File: deep-dive/language_model_clients/lm_local_models/MLC.md

# ChatModuleClient

!!! warning "This page is outdated and may not be fully accurate in DSPy 2.5"


## Prerequisites

1. Install the required packages using the following commands:
   
   ```shell
   pip install --no-deps --pre --force-reinstall mlc-ai-nightly-cu118 mlc-chat-nightly-cu118 -f https://mlc.ai/wheels
   pip install transformers
   git lfs install
   ```
   
   Adjust the pip wheels according to your OS/platform by referring to the provided commands in [MLC packages](https://mlc.ai/package/).

## Running MLC Llama-2 models

1. Create a directory for prebuilt models:

   ```shell
   mkdir -p dist/prebuilt
   ```
   
2. Clone the necessary libraries from the repository:

   ```shell
   git clone https://github.com/mlc-ai/binary-mlc-llm-libs.git dist/prebuilt/lib
   cd dist/prebuilt
   ```
   
3. Choose a Llama-2 model from [MLC LLMs](https://huggingface.co/mlc-ai) and clone the model repository:

   ```shell
   git clone https://huggingface.co/mlc-ai/mlc-chat-Llama-2-7b-chat-hf-q4f16_1
   ```

4. Initialize the `ChatModuleClient` within your program with the desired parameters. Here's an example call:

   ```python
   llama = dspy.ChatModuleClient(model='dist/prebuilt/mlc-chat-Llama-2-7b-chat-hf-q4f16_1', model_path='dist/prebuilt/lib/Llama-2-7b-chat-hf-q4f16_1-cuda.so')
   ```
Please refer to the [official MLC repository](https://github.com/mlc-ai/mlc-llm) for more detailed information and [documentation](https://mlc.ai/mlc-llm/docs/get_started/try_out.html).


## End of deep-dive/language_model_clients/lm_local_models/MLC.md

---


## File: deep-dive/language_model_clients/lm_local_models/HFClientVLLM.md

# HFClient vLLM

!!! warning "This page is outdated and may not be fully accurate in DSPy 2.5"


## Prerequisites

Follow these steps to set up the vLLM Server:

1. Build the server from source by following the instructions provided in the [Build from Source guide](https://vllm.readthedocs.io/en/latest/getting_started/installation.html#build-from-source).

2. Start the server by running the following command, and specify your desired model, host, and port using the appropriate arguments. The default server address is http://localhost:8000.

Example command:

```bash
   python -m vllm.entrypoints.openai.api_server --model mosaicml/mpt-7b --port 8000
```

This command will start the server and make it accessible at `http://localhost:8080`.

## Using the vLLM Client

After setting up the vLLM server and ensuring that it displays "Connected" when it's running, you can interact with it using the `HFClientVLLM`.

Initialize the `HFClientVLLM` within your program with the desired parameters. Here is an example call:

```python
   vllm_mpt = dspy.HFClientVLLM(model="mosaicml/mpt-7b", port=8000, url="http://localhost")
```
Customize the `model`, `port`, `url`, and `max_tokens` according to your requirements. The `model` parameter should be set to the specific Hugging Face model ID you wish to use.

Please refer to the [official vLLM repository](https://github.com/vllm-project/vllm) for more detailed information and documentation.

## Sending Requests via vLLM Client

1) _**Recommended**_ Configure default LM using `dspy.configure`.

This allows you to define programs in DSPy and simply call modules on your input fields, having DSPy internally call the prompt on the configured LM.

```python
dspy.configure(lm=vllm_mpt)

#Example DSPy CoT QA program
qa = dspy.ChainOfThought('question -> answer')

response = qa(question="What is the capital of Paris?") #Prompted to vllm_mpt
print(response.answer)
```

2) Generate responses using the client directly.

```python
response = vllm_mpt._generate(prompt='What is the capital of Paris?')
print(response)
```

## Under the Hood

### `__init__(self, model, port, url="http://localhost", **kwargs)`

The constructor initializes the `HFModel` base class to support the handling of prompting models, configuring the client for communicating with the hosted vLLM server to generate requests. This requires the following parameters:

- `model` (_str_): ID of model connected to the vLLM server.
- `port` (_int_): Port for communicating to the vLLM server. 
- `url` (_str_): Base URL of hosted vLLM server. This will often be `"http://localhost"`.
- `**kwargs`: Additional keyword arguments to configure the vLLM client.

Example of the vLLM constructor:

```python
class HFClientVLLM(HFModel):
    def __init__(self, model, port, url="http://localhost", **kwargs):
```

### `_generate(self, prompt, **kwargs) -> dict`

**Parameters:**
- `prompt` (_str_): Prompt to send to model hosted on vLLM server.
- `**kwargs`: Additional keyword arguments for completion request.

**Returns:**
- `dict`: dictionary with `prompt` and list of response `choices`.

Internally, the method handles the specifics of preparing the request prompt and corresponding payload to obtain the response. 

After generation, the method parses the JSON response received from the server and retrieves the output through `json_response["choices"]` and stored as the `completions` list.

Lastly, the method constructs the response dictionary with two keys: the original request `prompt` and `choices`, a list of dictionaries representing generated completions with the key `text` holding the response's generated text.


## End of deep-dive/language_model_clients/lm_local_models/HFClientVLLM.md

---


## File: deep-dive/modules/retrieve.md

# Retrieve

!!! warning "This page is outdated and may not be fully accurate in DSPy 2.5"

## Background
DSPy supports retrieval through the Retrieve module that serves to process user queries and output relevant passages from retrieval corpuses. This module ties in with the DSPy-supported Retrieval Model Clients which are retrieval servers that users can utilize to retrieve information for information retrieval tasks.

## Instantiating Retrieve

Retrieve is simply instantiate by a user-defined `k` number of retrieval passages to return for a query.

```python
class Retrieve(Parameter):
    def __init__(self, k=3):
        self.stage = random.randbytes(8).hex()
        self.k = k
```

Additionally, configuring a Retrieval model client or server through `dspy.configure` allows for user retrieval in DSPy programs through internal calls from Retrieve. 

```python
#Example Usage

#Define a retrieval model server to send retrieval requests to
colbertv2_wiki17_abstracts = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')

#Configure retrieval server internally
dspy.settings.configure(rm=colbertv2_wiki17_abstracts)

#Define Retrieve Module
retriever = dspy.Retrieve(k=3)
```

### Under the Hood

Retrieve makes use of the internally configured retriever to send a single query or list of multiple queries to determine the top-k relevant passages. The module queries the retriever for each provided query, accumulating scores (or probabilities if the `by_prob` arg is set) for each passage and returns the passages sorted by their cumulative scores or probabilities. 

The Retrieve module can also integrate a reranker if this is configured, in which case, the reranker re-scores the retrieved passages based on their relevance to the quer, improving accuracy of the results. 

### Tying it All Together

We can now call the Retrieve module on a sample query or list of queries and observe the top-K relevant passages.

```python
query='When was the first FIFA World Cup held?'

# Call the retriever on a particular query.
topK_passages = retriever(query).passages

print(f"Top {retriever.k} passages for question: {query} \n", '-' * 30, '\n')

for idx, passage in enumerate(topK_passages):
    print(f'{idx+1}]', passage, '\n')
```

```
Top 3 passages for question: When was the first FIFA World Cup held? 
 ------------------------------ 

1] History of the FIFA World Cup | The FIFA World Cup was first held in 1930, when FIFA president Jules Rimet [...]. 

2] 1950 FIFA World Cup | The 1950 FIFA World Cup, held in Brazil from 24 June to 16 July 1950, [...]. 

3] 1970 FIFA World Cup | The 1970 FIFA World Cup was the ninth FIFA World Cup, the quadrennial [...].
```


## End of deep-dive/modules/retrieve.md

---


## File: deep-dive/modules/multi-chain-comparison.md

# dspy.MultiChainComparison

!!! warning "This page is outdated and may not be fully accurate in DSPy 2.5"


## Constructor

The constructor initializes the `MultiChainComparison` class and sets up its attributes. It inherits from the `Predict` class and adds specific functionality for multiple chain comparisons.

The class incorporates multiple student attempt reasonings and concludes with the selected best reasoning path out of the available attempts.

```python
from .predict import Predict
from ..primitives.program import Module

import dsp

class MultiChainComparison(Module):
    def __init__(self, signature, M=3, temperature=0.7, **config):
        super().__init__()

        self.M = M
        signature = Predict(signature).signature
        *keys, last_key = signature.kwargs.keys()

        extended_kwargs = {key: signature.kwargs[key] for key in keys}

        for idx in range(M):
            candidate_type = dsp.Type(prefix=f"Student Attempt #{idx+1}:", desc="${reasoning attempt}")
            extended_kwargs.update({f'reasoning_attempt_{idx+1}': candidate_type})
        
        rationale_type = dsp.Type(prefix="Accurate Reasoning: Thank you everyone. Let's now holistically", desc="${corrected reasoning}")
        extended_kwargs.update({'rationale': rationale_type, last_key: signature.kwargs[last_key]})

        signature = dsp.Template(signature.instructions, **extended_kwargs)
        self.predict = Predict(signature, temperature=temperature, **config)
        self.last_key = last_key
```

**Parameters:**
- `signature` (_Any_): Signature of predictive model.
- `M` (_int_, _optional_): Number of student reasoning attempts. Defaults to `3`.
- `temperature` (_float_, _optional_): Temperature parameter for prediction. Defaults to `0.7`.
- `**config` (_dict_): Additional configuration parameters for model.

## Method

### `forward(self, completions, **kwargs)`

This method aggregates all the student reasoning attempts and calls the predict method with extended signatures to get the best reasoning.

**Parameters:**
- `completions`: List of completion objects which include student reasoning attempts.
- `**kwargs`: Additional keyword arguments.

**Returns:**
- The result of the `predict` method for the best reasoning.

### Examples

```python
class BasicQA(dspy.Signature):
    """Answer questions with short factoid answers."""
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")

# Example completions generated by a model for reference
completions = [
    dspy.Prediction(rationale="I recall that during clear days, the sky often appears this color.", answer="blue"),
    dspy.Prediction(rationale="Based on common knowledge, I believe the sky is typically seen as this color.", answer="green"),
    dspy.Prediction(rationale="From images and depictions in media, the sky is frequently represented with this hue.", answer="blue"),
]

# Pass signature to MultiChainComparison module
compare_answers = dspy.MultiChainComparison(BasicQA)

# Call the MultiChainComparison on the completions
question = 'What is the color of the sky?'
final_pred = compare_answers(completions, question=question)

print(f"Question: {question}")
print(f"Final Predicted Answer (after comparison): {final_pred.answer}")
print(f"Final Rationale: {final_pred.rationale}")
```


## End of deep-dive/modules/multi-chain-comparison.md

---


## File: deep-dive/modules/chain-of-thought-with-hint.md

---
sidebar_position: 2
---

# ChainOfThoughtWithHint

!!! warning "This page is outdated and may not be fully accurate in DSPy 2.5"

This class builds upon the `ChainOfThought` class by introducing an additional input field to provide hints for reasoning. The inclusion of a hint allows for a more directed problem-solving process, which can be especially useful in complex scenarios.


`ChainOfThoughtWithHint` is instantiated with a user-defined DSPy Signature, and the inclusion of a `hint` argument that takes in a string-form phrase to provide a hint within the prompt template.

Let's take a look at an example:

```python
class BasicQA(dspy.Signature):
    """Answer questions with short factoid answers."""
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")

#Pass signature to ChainOfThought module
generate_answer = dspy.ChainOfThoughtWithHint(BasicQA)

# Call the predictor on a particular input alongside a hint.
question='What is the color of the sky?'
hint = "It's what you often see during a sunny day."
pred = generate_answer(question=question, hint=hint)

print(f"Question: {question}")
print(f"Predicted Answer: {pred.answer}")
```


## End of deep-dive/modules/chain-of-thought-with-hint.md

---


## File: deep-dive/modules/program-of-thought.md

---
sidebar_position: 2
---

# Program of Thought

!!! warning "This page is outdated and may not be fully accurate in DSPy 2.5"


## Background

DSPy supports the Program of Thought (PoT) prompting technique, integrating an advanced module capable of problem-solving with program execution capabilities. PoT builds upon Chain of Thought by translating reasoning steps into executable programming language statements through iterative refinement. This enhances the accuracy of the output and self-corrects errors within the generated code. Upon completing these iterations, PoT delegates execution to a program interpreter. Currently, this class supports the generation and execution of Python code.

## Instantiating Program of Thought 

Program of Thought is instantiated based on a user-defined DSPy Signature, which can take a simple form such as `input_fields -> output_fields`. Users can also specify a `max_iters` to set the maximum number of iterations for the self-refinement process of the generated code. The default value is 3 iterations. Once the maximum iterations are reached, PoT will produce the final output.

```python
import dsp
import dspy
from ..primitives.program import Module
from ..primitives.python_interpreter import PythonInterpreter
import re

class ProgramOfThought(Module):
    def __init__(self, signature, max_iters=3):
        ...
```

```python
#Example Usage

#Define a simple signature for basic question answering
generate_answer_signature = dspy.Signature("question -> answer")
generate_answer_signature.attach(question=("Question:", "")).attach(answer=("Answer:", "often between 1 and 5 words"))

# Pass signature to ProgramOfThought Module
pot = dspy.ProgramOfThought(generate_answer_signature)
```

## Under the Hood

Program of Thought operates in 3 key modes: generate, regenerate, and answer. Each mode is a Chain of Thought module encapsulating signatures and instructions for each mode's individual purpose. These are dynamically created as the PoT iterations complete, accounting for the user's input and output fields internally.

**Generate mode:**

Initiates the generation of Python code with the signature `(question -> generated_code)` for the initial code generation.

**Regenerate mode:**

Used for refining the generation of Python code, considering the previously generated code and existing errors, with the signature `(question, previous_code, error -> generated_code)`. If errors persist past the maximum iterations, the user is alerted and the output is returned as `None`.

**Answer mode:**

Executes the last stored generated code and outputs the final answer, with the signature `(question, final_generated_code, code_output -> answer)`.

**Key internals:**
- **Code Parsing:**
    Program of Thought internally processes each code generation as a string and filters out extraneous bits to ensure the code block conforms to executable Python syntax. If the code is empty or does not match these guidelines, the parser returns an error string, signaling the PoT process for regeneration.
- **Code Execution:**
    Program of Thought relies on a sandboxed environment Python interpreter using Deno and Pyodide adapted by [this tutorial](https://til.simonwillison.net/deno/pyodide-sandbox). This adaptation is present in [DSPy primitives](https://github.com/stanfordnlp/dspy/blob/main/dspy/primitives/python_interpreter.py).

## Tying It All Together
Using ProgramOfThought mirrors the simplicity of the base `Predict` and `ChainOfThought` modules. Here is an example call:

```python
#Call the ProgramOfThought module on a particular input
question = 'Sarah has 5 apples. She buys 7 more apples from the store. How many apples does Sarah have now?'
result = pot(question=question)

print(f"Question: {question}")
print(f"Final Predicted Answer (after ProgramOfThought process): {result.answer}")
```
```
Question: Sarah has 5 apples. She buys 7 more apples from the store. How many apples does Sarah have now?
Final Predicted Answer (after ProgramOfThought process): 12
```

Let's take a peek at how ProgramOfThought functioned internally by inspecting its history, up to maximum iterations (+ 1 to view the final generation). (This assumes the initial DSPy setup and configurations of LMs and RMs). 

`lm.inspect_history(n=4)`

```
You will be given `question` and you will respond with `generated_code`.
Generating executable Python code that programmatically computes the correct `generated_code`.
After you're done with the computation, make sure the last line in your code evaluates to the correct value for `generated_code`.

---

Follow the following format.

Question: 
Reasoning: Let's think step by step in order to ${produce the generated_code}. We ...
Code: python code that answers the question

---

Question: Sarah has 5 apples. She buys 7 more apples from the store. How many apples does Sarah have now?
Reasoning: Let's think step by step in order to produce the generated_code. We start with the initial number of apples that Sarah has, which is 5. Then, we add the number of apples she buys from the store, which is 7. Finally, we compute the total number of apples Sarah has by adding these two quantities together.
Code: 

apples_initial = 5
apples_bought = 7

total_apples = apples_initial + apples_bought
total_apples

Given the final code `question`, `final_generated_code`, `code_output`, provide the final `answer`.

---

Follow the following format.

Question: 

Code: python code that answers the question

Code Output: output of previously-generated python code

Reasoning: Let's think step by step in order to ${produce the answer}. We ...

Answer: often between 1 and 5 words

---

Question: Sarah has 5 apples. She buys 7 more apples from the store. How many apples does Sarah have now?

Code:
apples_initial = 5
apples_bought = 7

total_apples = apples_initial + apples_bought
total_apples

Code Output: 12

Reasoning: Let's think step by step in order to produce the answer. We start with the initial number of apples Sarah has, which is 5. Then, we add the number of apples she bought, which is 7. The total number of apples Sarah has now is 12.

Answer: 12

```


## End of deep-dive/modules/program-of-thought.md

---


## File: deep-dive/modules/guide.md

# DSPy Modules

!!! warning "This page is outdated and may not be fully accurate in DSPy 2.5"


[<img align="center" src="https://colab.research.google.com/assets/colab-badge.svg" />](https://colab.research.google.com/github/stanfordnlp/dspy/blob/main/docs/guides/modules.ipynb)

## Quick Recap

This guide assumes you followed the [intro tutorial](https://colab.research.google.com/github/stanfordnlp/dspy/blob/main/intro.ipynb) to build your first few DSPy programs.

Remember that **DSPy program** is just Python code that calls one or more **DSPy modules**, like `dspy.Predict` or `dspy.ChainOfThought`, to use LMs.

## 1) What is a DSPy Module?

A **DSPy module** is a building block for programs that use LMs.

- Each built-in module abstracts a **prompting technique** (like chain of thought or ReAct). Crucially, they are generalized to handle any [DSPy Signature](/building-blocks/2-signatures).

- A DSPy module has **learnable parameters** (i.e., the little pieces comprising the prompt and the LM weights) and can be invoked (called) to process inputs and return outputs.

- Multiple modules can be composed into bigger modules (programs). DSPy modules are inspired directly by NN modules in PyTorch, but applied to LM programs.

## 2) What DSPy Modules are currently built-in?

1. **[`dspy.Predict`](/deep-dive/modules/predict)**:

2. **[`dspy.ChainOfThought`](/deep-dive/modules/chain-of-thought)**: 

3. **[`dspy.ProgramOfThought`](/deep-dive/modules/program-of-thought)**:

4. **[`dspy.ReAct`](/deep-dive/modules/ReAct)**:

5. **[`dspy.MultiChainComparison`](/deep-dive/modules/multi-chain-comparison)**:

## 3) How do I use a built-in module, like `dspy.Predict` or `dspy.ChainOfThought`?

Let's start with the most fundamental one, `dspy.Predict`. Internally, all of the others are just built using it!

We'll assume you are already at least a little familiar with [DSPy signatures](/building-blocks/2-signatures), which are declarative specs for defining the behavior of any module we use in DSPy.
To use a module, we first **declare** it by giving it a signature. Then we **call** the module with the input arguments, and extract the output fields!


```python
sentence = "it's a charming and often affecting journey."  # example from the SST-2 dataset.

# 1) Declare with a signature.
classify = dspy.Predict('sentence -> sentiment')

# 2) Call with input argument(s). 
response = classify(sentence=sentence)

# 3) Access the output.
print(response.sentiment)
```

```text
Positive
```
    

When we declare a module, we can pass configuration keys to it.

Below, we'll pass `n=5` to request five completions. We can also pass `temperature` or `max_len`, etc.

Let's use `dspy.ChainOfThought`. In many cases, simply swapping `dspy.ChainOfThought` in place of `dspy.Predict` improves quality.


```python
question = "What's something great about the ColBERT retrieval model?"

# 1) Declare with a signature, and pass some config.
classify = dspy.ChainOfThought('question -> answer', n=5)

# 2) Call with input argument.
response = classify(question=question)

# 3) Access the outputs.
response.completions.answer
```

```text
['One great thing about the ColBERT retrieval model is its superior efficiency and effectiveness compared to other models.',
 'Its ability to efficiently retrieve relevant information from large document collections.',
 'One great thing about the ColBERT retrieval model is its superior performance compared to other models and its efficient use of pre-trained language models.',
 'One great thing about the ColBERT retrieval model is its superior efficiency and accuracy compared to other models.',
 'One great thing about the ColBERT retrieval model is its ability to incorporate user feedback and support complex queries.']
```


Let's discuss the output object here.

The `dspy.ChainOfThought` module will generally inject a `reasoning` before the output field(s) of your signature.

Let's inspect the (first) reasoning and answer!


```python
print(f"Reasoning: {response.reasoning}")
print(f"Answer: {response.answer}")
```

    Reasoning: ColBERT (Contextualized Late Interaction over BERT) is a retrieval model that effectively combines the strengths of dense retrieval and traditional BM25 methods. One of its great features is that it allows for efficient and scalable retrieval by using late interaction techniques, which enables the model to leverage the contextual embeddings generated by BERT while still maintaining a fast retrieval speed. This means that it can handle large document collections more effectively than many other models, providing both high relevance in search results and efficiency in processing time.
    Answer: A great feature of the ColBERT retrieval model is its ability to efficiently combine contextualized embeddings from BERT with a late interaction mechanism, allowing for scalable and high-performance document retrieval.

    

This is accessible whether we request one or many completions.

We can also access the different completions as a list of `Prediction`s or as several lists, one for each field.


```python
response.completions[3].reasoning == response.completions.reasoning[3]
```

```text
True
```

## 4) How do I use more complex built-in modules?

The others are very similar, `dspy.ReAct` and `dspy.ProgramOfThought` etc. They mainly change the internal behavior with which your signature is implemented!

Check out further examples in [each module's respective guide]().

## 5) How do I compose multiple modules into a bigger program?

DSPy is just Python code that uses modules in any control flow you like. (There's some magic internally at `compile` time to trace your LM calls.)

What this means is that, you can just call the modules freely. No weird abstractions for chaining calls.

This is basically PyTorch's design approach for define-by-run / dynamic computation graphs. Refer to the intro tutorials for examples.


## End of deep-dive/modules/guide.md

---


## File: deep-dive/modules/react.md

---
sidebar_position: 2
---

# ReAct

!!! warning "This page is outdated and may not be fully accurate in DSPy 2.5"


## Background

DSPy supports ReAct, an LLM agent designed to tackle complex tasks in an interactive fashion. ReAct is composed of an iterative loop of interpretation, decision and action-based activities ("Thought, Action, and Observation") based on an evolving set of input and output fields. Through this real-time iterative approach, the ReAct agent can both analyze and adapt to its responses over time as new information becomes available.

## Instantiating ReAct 

To instantiate the ReAct module, define and pass in a DSPy Signature. 

```python
# Define a simple signature for basic question answering
class BasicQA(dspy.Signature):
    """Answer questions with short factoid answers."""
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")

# Pass signature to ReAct module
react_module = dspy.ReAct(BasicQA)
```

## Under the Hood

### ReAct Cycle

ReAct operates under a dynamic signature integration process, accounting for the signature inputs and performing the Thoughts, Action, Observation cycle to respond with the signature outputs. Thoughts (or reasoning) lead to Actions (such as queries or activities). These Actions then result in Observations (like results or responses), which subsequently feedback into the next Thought.

This cycle is maintained for a predefined number of iterations, specified by `max_iters`. The default value for the Thought-Action-Observation cycle is 5 iterations. Once the maximum iterations are reached, React will return the final output if the Action has finished `(Finish[answer])` or an empty string to indicate the agent could not determine a final output.

!!! caution
    Currently, ReAct supports only one output field in its signature. We plan to expand this in future developments.

### ReAct Tools

Tools in ReAct can shape the agent's interaction and response mechanisms, and DSPy ensures this customizability by allowing users to pass in their toolsets tailored for their task scenarios. The default tool is the `dspy.Retrieve`
module (serving to retrieve information from Retrieval Models during the Action step) with default `num_results=3`, and these can be passed as arguments to the initialization of the ReAct module.



## Tying It All Together
Using ReAct mirrors the simplicity of the base `Predict` and `ChainOfThought` modules. Here is an example call:

```python
# Call the ReAct module on a particular input
question = 'Aside from the Apple Remote, what other devices can control the program Apple Remote was originally designed to interact with?'
result = react_module(question=question)

print(f"Question: {question}")
print(f"Final Predicted Answer (after ReAct process): {result.answer}")
```
```
Question: Aside from the Apple Remote, what other devices can control the program Apple Remote was originally designed to interact with?
Final Predicted Answer (after ReAct process): The Apple Remote and the Siri Remote can control the Front Row media program.
```

Let's take a peek at how ReAct functioned internally by inspecting its history, up to maximum iterations. (This assumes the initial DSPy setup and configurations of LMs and RMs). 

`lm.inspect_history(n=3)`

```
-------------------------Step 1---------------------------------
You will be given `question` and you will respond with `answer`.

To do this, you will interleave Thought, Action, and Observation steps.

Thought can reason about the current situation, and Action can be the following types:

(1) Search[query], which takes a search query and returns one or more potentially relevant passages from a corpus
(2) Finish[answer], which returns the final `answer` and finishes the task

---

Follow the following format.

Question: ${question}
Thought 1: next steps to take based on last observation
Action 1: always either Search[query] or, when done, Finish[answer]

---

Question: Aside from the Apple Remote, what other devices can control the program Apple Remote was originally designed to interact with?
Thought 1: I need to find out what program the Apple Remote was originally designed to interact with.
Action 1: Search["program Apple Remote was originally designed to interact with"]

-------------------------Step 2---------------------------------
You will be given `question` and you will respond with `answer`.

To do this, you will interleave Thought, Action, and Observation steps.

Thought can reason about the current situation, and Action can be the following types:

(1) Search[query], which takes a search query and returns one or more potentially relevant passages from a corpus
(2) Finish[answer], which returns the final `answer` and finishes the task

---

Follow the following format.

Question: ${question}

Thought 1: next steps to take based on last observation

Action 1: always either Search[query] or, when done, Finish[answer]

Observation 1: observations based on action

Thought 2: next steps to take based on last observation

Action 2: always either Search[query] or, when done, Finish[answer]

---

Question: Aside from the Apple Remote, what other devices can control the program Apple Remote was originally designed to interact with?

[Thought 1, Action 1]

Observation 1:
[1] «Apple Remote | The Apple Remote is a remote control device released [...].»
[2] «ITunes Remote | iTunes Remote (also known simply as Remote) is a software [...].»
[3] «Siri Remote | The Siri Remote is a remote control device released in tandem [...]»

Thought 2: I have found that the Apple Remote was originally designed to interact with the Front Row media program on the iSight iMac G5. I should search for other devices that can control this program.

Action 2: Search["devices that can control Front Row media program"]

Observation 2:
[1] «Apple Remote» - The Apple Remote is the primary control mechanism for the Front Row media program [...].
[2] «Siri Remote» - The Siri Remote is the successor of the Apple Remote and is designed [...].

-------------------------Step 3---------------------------------

You will be given `question` and you will respond with `answer`.

To do this, you will interleave Thought, Action, and Observation steps.

Thought can reason about the current situation, and Action can be the following types:

(1) Search[query], which takes a search query and returns one or more potentially relevant passages from a corpus
(2) Finish[answer], which returns the final `answer` and finishes the task

---

Follow the following format.

Question: ${question}

Thought 1: next steps to take based on last observation

Action 1: always either Search[query] or, when done, Finish[answer]

Observation 1: observations based on action

Thought 2: next steps to take based on last observation

Action 2: always either Search[query] or, when done, Finish[answer]

Observation 2: observations based on action

Thought 3: next steps to take based on last observation

Action 3: always either Search[query] or, when done, Finish[answer]

---

Question: Aside from the Apple Remote, what other devices can control the program Apple Remote was originally designed to interact with?

[Thought 1, Action 1, Observation 1, Thought 2, Action 2]

Observation 2:
[1] «Front Row (software) | Front Row is a discontinued media center [...].»
[2] «Apple Remote | The Apple Remote is a remote control device [...].»
[3] «Media player (software) | A media player is a computer program for [...].»

Thought 3: I have found that the Apple Remote and the Siri Remote can control the Front Row media program. I should provide this information as the answer.

Action 3: Finish[The Apple Remote and the Siri Remote can control the Front Row media program.]

Answer: The Apple Remote and the Siri Remote can control the Front Row media program.
```


## End of deep-dive/modules/react.md

---


## File: deep-dive/modules/predict.md

# dspy.Predict

!!! warning "This page is outdated and may not be fully accurate in DSPy 2.5"


## Constructor

The constructor initializes the `Predict` class and sets up its attributes, taking in the `signature` and additional config options. If the `signature` is a string, it processes the input and output fields, generates instructions, and creates a template for the specified `signature` type.

```python
class Predict(Parameter):
    def __init__(self, signature, **config):
        self.stage = random.randbytes(8).hex()
        self.signature = signature
        self.config = config
        self.reset()

        if isinstance(signature, str):
            inputs, outputs = signature.split("->")
            inputs, outputs = inputs.split(","), outputs.split(",")
            inputs, outputs = [field.strip() for field in inputs], [field.strip() for field in outputs]

            assert all(len(field.split()) == 1 for field in (inputs + outputs))

            inputs_ = ', '.join([f"`{field}`" for field in inputs])
            outputs_ = ', '.join([f"`{field}`" for field in outputs])

            instructions = f"""Given the fields {inputs_}, produce the fields {outputs_}."""

            inputs = {k: InputField() for k in inputs}
            outputs = {k: OutputField() for k in outputs}

            for k, v in inputs.items():
                v.finalize(k, infer_prefix(k))
            
            for k, v in outputs.items():
                v.finalize(k, infer_prefix(k))

            self.signature = dsp.Template(instructions, **inputs, **outputs)
```

**Parameters:**
- `signature` (_Any_): Signature of predictive model.
- `**config` (_dict_): Additional configuration parameters for model.

## Method

### `__call__(self, **kwargs)`

This method serves as a wrapper for the `forward` method. It allows making predictions using the `Predict` class by providing keyword arguments.

**Parameters:**
- `**kwargs`: Keyword arguments required for prediction.

**Returns:**
- The result of `forward` method.

## Examples

```python
#Define a simple signature for basic question answering
class BasicQA(dspy.Signature):
    """Answer questions with short factoid answers."""
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")

#Pass signature to Predict module
generate_answer = dspy.Predict(BasicQA)

# Call the predictor on a particular input.
question='What is the color of the sky?'
pred = generate_answer(question=question)

print(f"Question: {question}")
print(f"Predicted Answer: {pred.answer}")
```


## End of deep-dive/modules/predict.md

---


## File: deep-dive/modules/chain-of-thought.md

# dspy.ChainOfThought

!!! warning "This page is outdated and may not be fully accurate in DSPy 2.5"


### Constructor

The constructor initializes the `ChainOfThought` class and sets up its attributes. It inherits from the `Predict` class and adds specific functionality for chain of thought processing. 

Internally, the class initializes the `activated` attribute to indicate if chain of thought processing has been selected. It extends the `signature` to include additional reasoning steps and an updated `rationale_type` when chain of thought processing is activated.

```python
class ChainOfThought(Predict):
    def __init__(self, signature, rationale_type=None, activated=True, **config):
        super().__init__(signature, **config)

        self.activated = activated

        self.signature = signature = ensure_signature(signature)
        *_keys, last_key = signature.output_fields.keys()

        prefix = "Reasoning: Let's think step by step in order to"

        if isinstance(dspy.settings.lm, dspy.LM):
            desc = "${reasoning}"
        elif dspy.settings.experimental:
            desc = "${produce the output fields}. We ..."
        else:
            # For dspy <2.5
            desc = f"${{produce the {last_key}}}. We ..."

        rationale_type = rationale_type or dspy.OutputField(prefix=prefix, desc=desc)

        # Add "rationale" field to the output signature.
        if isinstance(dspy.settings.lm, dspy.LM) or dspy.settings.experimental:
            extended_signature = signature.prepend("reasoning", rationale_type, type_=str)
        else:
            # For dspy <2.5
            extended_signature = signature.prepend("rationale", rationale_type, type_=str)
        
        self._predict = dspy.Predict(extended_signature, **config)
        self._predict.extended_signature = extended_signature
```

**Parameters:**
- `signature` (_Any_): Signature of predictive model.
- `rationale_type` (_dspy.OutputField_, _optional_): Rationale type for reasoning steps. Defaults to `None`.
- `activated` (_bool_, _optional_): Flag for activated chain of thought processing. Defaults to `True`.
- `**config` (_dict_): Additional configuration parameters for model.

### Method

#### `forward(self, **kwargs)`

This method extends the parent `Predict` class' forward pass while updating the signature when chain of thought reasoning is activated or if the language model is a GPT3 model.

**Parameters:**
- `**kwargs`: Keyword arguments required for prediction.

**Returns:**
- The result of the `forward` method.

### Examples

```python
#Define a simple signature for basic question answering
class BasicQA(dspy.Signature):
    """Answer questions with short factoid answers."""
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")

#Pass signature to ChainOfThought module
generate_answer = dspy.ChainOfThought(BasicQA)

# Call the predictor on a particular input.
question='What is the color of the sky?'
pred = generate_answer(question=question)

print(f"Question: {question}")
print(f"Predicted Answer: {pred.answer}")
```

The following example shows how to specify your custom rationale. Here `answer` corresponds to the last key to produce, it may be different in your case. 

```python
#define a custom rationale
rationale_type = dspy.OutputField(
            prefix="Reasoning: Let's think step by step in order to",
            desc="${produce the answer}. We ...",
        )
#Pass signature to ChainOfThought module
generate_answer = dspy.ChainOfThought(BasicQA, rationale_type=rationale_type)
```

## End of deep-dive/modules/chain-of-thought.md

---


## File: production/index.md

# Using DSPy in Production

<div class="grid cards" style="text-align: left;" markdown>

- :material-earth:{ .lg .middle } __Real-World Use Cases__

    ---

    DSPy is deployed in production by many enterprises and startups. Explore real-world case studies.

    [:octicons-arrow-right-24: Use Cases](../dspy-usecases.md)

- :material-magnify-expand:{ .lg .middle } __Monitoring & Observability__

    ---

    Monitor your DSPy programs using **MLflow Tracing**, based on OpenTelemetry.

    [:octicons-arrow-right-24: Set Up Observability](../tutorials/observability/#tracing)

- :material-ab-testing: __Reproducibility__

    ---

    Log programs, metrics, configs, and environments for full reproducibility with DSPy's native MLflow integration.

    [:octicons-arrow-right-24: MLflow Integration](https://mlflow.org/docs/latest/llms/dspy/index.html)

- :material-rocket-launch: __Deployment__

    ---

    When it's time to productionize, deploy your application easily with DSPy's integration with MLflow Model Serving.

    [:octicons-arrow-right-24: Deployment Guide](../tutorials/deployment)

- :material-arrow-up-right-bold: __Scalability__

    ---

    DSPy is designed with thread-safety in mind and offers native asynchronous execution support for high-throughput environments.

    [:octicons-arrow-right-24: Async Program](../api/utils/asyncify.md)

- :material-alert-rhombus: __Guardrails & Controllability__

    ---

    DSPy's **Signatures**, **Modules**, and **Optimizers** help you control and guide LM outputs.

    [:octicons-arrow-right-24: Learn Signature](../learn/programming/signatures.md)

</div>


## End of production/index.md

---


## File: api/index.md

# API Reference

Welcome to the DSPy API reference documentation. This section provides detailed information about DSPy's classes, modules, and functions.

## End of api/index.md

---


## File: api/tools/ColBERTv2.md

# dspy.ColBERTv2

::: dspy.ColBERTv2
    handler: python
    options:
        members:
            - __call__
        show_source: true
        show_undocumented_members: true
        show_root_heading: true
        show_inherited_members: true
        heading_level: 2
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false


## End of api/tools/ColBERTv2.md

---


## File: api/tools/Embeddings.md

# dspy.retrievers.Embeddings

::: dspy.retrievers.Embeddings
    handler: python
    options:
        members:
            - __call__
            - forward
        show_source: true
        show_undocumented_members: true
        show_root_heading: true
        show_inherited_members: true
        heading_level: 2
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false


## End of api/tools/Embeddings.md

---


## File: api/tools/PythonInterpreter.md

# dspy.PythonInterpreter

::: dspy.PythonInterpreter
    handler: python
    options:
        members:
            - __call__
            - execute
            - shutdown
        show_source: true
        show_undocumented_members: true
        show_root_heading: true
        show_inherited_members: true
        heading_level: 2
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false


## End of api/tools/PythonInterpreter.md

---


## File: api/signatures/InputField.md

# dspy.InputField

::: dspy.InputField
    handler: python
    options:
        show_source: true
        show_undocumented_members: true
        show_root_heading: true
        show_inherited_members: true
        heading_level: 2
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false


## End of api/signatures/InputField.md

---


## File: api/signatures/Signature.md

# dspy.Signature

::: dspy.Signature
    handler: python
    options:
        members:
            - append
            - delete
            - dump_state
            - equals
            - insert
            - load_state
            - prepend
            - with_instructions
            - with_updated_fields
        show_source: true
        show_undocumented_members: true
        show_root_heading: true
        show_inherited_members: true
        heading_level: 2
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false


## End of api/signatures/Signature.md

---


## File: api/signatures/OutputField.md

# dspy.OutputField

::: dspy.OutputField
    handler: python
    options:
        show_source: true
        show_undocumented_members: true
        show_root_heading: true
        show_inherited_members: true
        heading_level: 2
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false


## End of api/signatures/OutputField.md

---


## File: api/utils/enable_logging.md

# dspy.enable_logging

::: dspy.enable_logging
    handler: python
    options:
        show_source: true
        show_undocumented_members: true
        show_root_heading: true
        show_inherited_members: true
        heading_level: 2
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false


## End of api/utils/enable_logging.md

---


## File: api/utils/StatusMessage.md

# dspy.utils.StatusMessage

::: dspy.utils.StatusMessage
    handler: python
    options:
        show_source: true
        show_undocumented_members: true
        show_root_heading: true
        show_inherited_members: true
        heading_level: 2
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false


## End of api/utils/StatusMessage.md

---


## File: api/utils/streamify.md

# dspy.streamify

::: dspy.streamify
    handler: python
    options:
        show_source: true
        show_undocumented_members: true
        show_root_heading: true
        show_inherited_members: true
        heading_level: 2
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false


## End of api/utils/streamify.md

---


## File: api/utils/load.md

# dspy.load

::: dspy.load
    handler: python
    options:
        show_source: true
        show_undocumented_members: true
        show_root_heading: true
        show_inherited_members: true
        heading_level: 2
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false


## End of api/utils/load.md

---


## File: api/utils/disable_litellm_logging.md

# dspy.disable_litellm_logging

::: dspy.disable_litellm_logging
    handler: python
    options:
        show_source: true
        show_undocumented_members: true
        show_root_heading: true
        show_inherited_members: true
        heading_level: 2
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false


## End of api/utils/disable_litellm_logging.md

---


## File: api/utils/enable_litellm_logging.md

# dspy.enable_litellm_logging

::: dspy.enable_litellm_logging
    handler: python
    options:
        show_source: true
        show_undocumented_members: true
        show_root_heading: true
        show_inherited_members: true
        heading_level: 2
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false


## End of api/utils/enable_litellm_logging.md

---


## File: api/utils/disable_logging.md

# dspy.disable_logging

::: dspy.disable_logging
    handler: python
    options:
        show_source: true
        show_undocumented_members: true
        show_root_heading: true
        show_inherited_members: true
        heading_level: 2
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false


## End of api/utils/disable_logging.md

---


## File: api/utils/StatusMessageProvider.md

# dspy.utils.StatusMessageProvider

::: dspy.utils.StatusMessageProvider
    handler: python
    options:
        members:
            - lm_end_status_message
            - lm_start_status_message
            - module_end_status_message
            - module_start_status_message
            - tool_end_status_message
            - tool_start_status_message
        show_source: true
        show_undocumented_members: true
        show_root_heading: true
        show_inherited_members: true
        heading_level: 2
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false


## End of api/utils/StatusMessageProvider.md

---


## File: api/utils/inspect_history.md

# dspy.inspect_history

::: dspy.inspect_history
    handler: python
    options:
        show_source: true
        show_undocumented_members: true
        show_root_heading: true
        show_inherited_members: true
        heading_level: 2
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false


## End of api/utils/inspect_history.md

---


## File: api/utils/asyncify.md

# dspy.asyncify

::: dspy.asyncify
    handler: python
    options:
        show_source: true
        show_undocumented_members: true
        show_root_heading: true
        show_inherited_members: true
        heading_level: 2
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false


## End of api/utils/asyncify.md

---


## File: api/models/Embedder.md

# dspy.Embedder

::: dspy.Embedder
    handler: python
    options:
        members:
            - __call__
        show_source: true
        show_undocumented_members: true
        show_root_heading: true
        show_inherited_members: true
        heading_level: 2
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false


## End of api/models/Embedder.md

---


## File: api/models/LM.md

# dspy.LM

::: dspy.LM
    handler: python
    options:
        members:
            - __call__
            - copy
            - dump_state
            - finetune
            - forward
            - infer_provider
            - inspect_history
            - kill
            - launch
            - update_global_history
        show_source: true
        show_undocumented_members: true
        show_root_heading: true
        show_inherited_members: true
        heading_level: 2
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false


## End of api/models/LM.md

---


## File: api/optimizers/Ensemble.md

# dspy.Ensemble

::: dspy.Ensemble
    handler: python
    options:
        members:
            - compile
            - get_params
        show_source: true
        show_undocumented_members: true
        show_root_heading: true
        show_inherited_members: true
        heading_level: 2
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false


## End of api/optimizers/Ensemble.md

---


## File: api/optimizers/KNNFewShot.md

# dspy.KNNFewShot

::: dspy.KNNFewShot
    handler: python
    options:
        members:
            - compile
            - get_params
        show_source: true
        show_undocumented_members: true
        show_root_heading: true
        show_inherited_members: true
        heading_level: 2
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false


## End of api/optimizers/KNNFewShot.md

---


## File: api/optimizers/BootstrapRS.md

# dspy.BootstrapRS

::: dspy.BootstrapRS
    handler: python
    options:
        members:
            - compile
            - get_params
        show_source: true
        show_undocumented_members: true
        show_root_heading: true
        show_inherited_members: true
        heading_level: 2
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false


## End of api/optimizers/BootstrapRS.md

---


## File: api/optimizers/MIPROv2.md

# dspy.MIPROv2

::: dspy.MIPROv2
    handler: python
    options:
        members:
            - compile
            - get_params
        show_source: true
        show_undocumented_members: true
        show_root_heading: true
        show_inherited_members: true
        heading_level: 2
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false


## End of api/optimizers/MIPROv2.md

---


## File: api/optimizers/KNN.md

# dspy.KNN

::: dspy.KNN
    handler: python
    options:
        members:
            - __call__
        show_source: true
        show_undocumented_members: true
        show_root_heading: true
        show_inherited_members: true
        heading_level: 2
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false


## End of api/optimizers/KNN.md

---


## File: api/optimizers/BootstrapFewShotWithRandomSearch.md

# dspy.BootstrapFewShotWithRandomSearch

::: dspy.BootstrapFewShotWithRandomSearch
    handler: python
    options:
        members:
            - compile
            - get_params
        show_source: true
        show_undocumented_members: true
        show_root_heading: true
        show_inherited_members: true
        heading_level: 2
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false


## End of api/optimizers/BootstrapFewShotWithRandomSearch.md

---


## File: api/optimizers/BootstrapFewShot.md

# dspy.BootstrapFewShot

::: dspy.BootstrapFewShot
    handler: python
    options:
        members:
            - compile
            - get_params
        show_source: true
        show_undocumented_members: true
        show_root_heading: true
        show_inherited_members: true
        heading_level: 2
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false


## End of api/optimizers/BootstrapFewShot.md

---


## File: api/optimizers/LabeledFewShot.md

# dspy.LabeledFewShot

::: dspy.LabeledFewShot
    handler: python
    options:
        members:
            - compile
            - get_params
        show_source: true
        show_undocumented_members: true
        show_root_heading: true
        show_inherited_members: true
        heading_level: 2
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false


## End of api/optimizers/LabeledFewShot.md

---


## File: api/optimizers/BootstrapFinetune.md

# dspy.BootstrapFinetune

::: dspy.BootstrapFinetune
    handler: python
    options:
        members:
            - compile
            - convert_to_lm_dict
            - finetune_lms
            - get_params
        show_source: true
        show_undocumented_members: true
        show_root_heading: true
        show_inherited_members: true
        heading_level: 2
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false


## End of api/optimizers/BootstrapFinetune.md

---


## File: api/optimizers/InferRules.md

# dspy.InferRules

::: dspy.InferRules
    handler: python
    options:
        members:
            - compile
            - evaluate_program
            - format_examples
            - get_params
            - get_predictor_demos
            - induce_natural_language_rules
            - update_program_instructions
        show_source: true
        show_undocumented_members: true
        show_root_heading: true
        show_inherited_members: true
        heading_level: 2
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false


## End of api/optimizers/InferRules.md

---


## File: api/optimizers/BetterTogether.md

# dspy.BetterTogether

::: dspy.BetterTogether
    handler: python
    options:
        members:
            - compile
            - get_params
        show_source: true
        show_undocumented_members: true
        show_root_heading: true
        show_inherited_members: true
        heading_level: 2
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false


## End of api/optimizers/BetterTogether.md

---


## File: api/optimizers/COPRO.md

# dspy.COPRO

::: dspy.COPRO
    handler: python
    options:
        members:
            - compile
            - get_params
        show_source: true
        show_undocumented_members: true
        show_root_heading: true
        show_inherited_members: true
        heading_level: 2
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false


## End of api/optimizers/COPRO.md

---


## File: api/primitives/Prediction.md

# dspy.Prediction

::: dspy.Prediction
    handler: python
    options:
        members:
            - copy
            - from_completions
            - get
            - inputs
            - items
            - keys
            - labels
            - toDict
            - values
            - with_inputs
            - without
        show_source: true
        show_undocumented_members: true
        show_root_heading: true
        show_inherited_members: true
        heading_level: 2
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false


## End of api/primitives/Prediction.md

---


## File: api/primitives/Example.md

# dspy.Example

::: dspy.Example
    handler: python
    options:
        members:
            - copy
            - get
            - inputs
            - items
            - keys
            - labels
            - toDict
            - values
            - with_inputs
            - without
        show_source: true
        show_undocumented_members: true
        show_root_heading: true
        show_inherited_members: true
        heading_level: 2
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false


## End of api/primitives/Example.md

---


## File: api/primitives/Image.md

# dspy.Image

::: dspy.Image
    handler: python
    options:
        members:
            - from_PIL
            - from_file
            - from_url
            - serialize_model
            - validate_input
        show_source: true
        show_undocumented_members: true
        show_root_heading: true
        show_inherited_members: true
        heading_level: 2
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false


## End of api/primitives/Image.md

---


## File: api/primitives/Tool.md

# dspy.Tool

::: dspy.Tool
    handler: python
    options:
        members:
            - __call__
        show_source: true
        show_undocumented_members: true
        show_root_heading: true
        show_inherited_members: true
        heading_level: 2
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false


## End of api/primitives/Tool.md

---


## File: api/primitives/History.md

# dspy.History

::: dspy.History
    handler: python
    options:
        show_source: true
        show_undocumented_members: true
        show_root_heading: true
        show_inherited_members: true
        heading_level: 2
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false


## End of api/primitives/History.md

---


## File: api/adapters/ChatAdapter.md

# dspy.ChatAdapter

::: dspy.ChatAdapter
    handler: python
    options:
        members:
            - __call__
            - format
            - format_conversation_history
            - format_fields
            - format_finetune_data
            - format_turn
            - parse
        show_source: true
        show_undocumented_members: true
        show_root_heading: true
        show_inherited_members: true
        heading_level: 2
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false


## End of api/adapters/ChatAdapter.md

---


## File: api/adapters/JSONAdapter.md

# dspy.JSONAdapter

::: dspy.JSONAdapter
    handler: python
    options:
        members:
            - __call__
            - format
            - format_conversation_history
            - format_fields
            - format_finetune_data
            - format_turn
            - parse
        show_source: true
        show_undocumented_members: true
        show_root_heading: true
        show_inherited_members: true
        heading_level: 2
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false


## End of api/adapters/JSONAdapter.md

---


## File: api/adapters/Adapter.md

# dspy.Adapter

::: dspy.Adapter
    handler: python
    options:
        members:
            - __call__
            - format
            - format_conversation_history
            - format_fields
            - format_finetune_data
            - format_turn
            - parse
        show_source: true
        show_undocumented_members: true
        show_root_heading: true
        show_inherited_members: true
        heading_level: 2
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false


## End of api/adapters/Adapter.md

---


## File: api/evaluation/Evaluate.md

# dspy.Evaluate

::: dspy.Evaluate
    handler: python
    options:
        members:
            - __call__
        show_source: true
        show_undocumented_members: true
        show_root_heading: true
        show_inherited_members: true
        heading_level: 2
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false


## End of api/evaluation/Evaluate.md

---


## File: api/evaluation/CompleteAndGrounded.md

# dspy.evaluate.CompleteAndGrounded

::: dspy.evaluate.CompleteAndGrounded
    handler: python
    options:
        members:
            - __call__
            - batch
            - deepcopy
            - dump_state
            - forward
            - get_lm
            - load
            - load_state
            - map_named_predictors
            - named_parameters
            - named_predictors
            - named_sub_modules
            - parameters
            - predictors
            - reset_copy
            - save
            - set_lm
        show_source: true
        show_undocumented_members: true
        show_root_heading: true
        show_inherited_members: true
        heading_level: 2
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false


## End of api/evaluation/CompleteAndGrounded.md

---


## File: api/evaluation/answer_exact_match.md

# dspy.evaluate.answer_exact_match

::: dspy.evaluate.answer_exact_match
    handler: python
    options:
        show_source: true
        show_undocumented_members: true
        show_root_heading: true
        show_inherited_members: true
        heading_level: 2
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false


## End of api/evaluation/answer_exact_match.md

---


## File: api/evaluation/answer_passage_match.md

# dspy.evaluate.answer_passage_match

::: dspy.evaluate.answer_passage_match
    handler: python
    options:
        show_source: true
        show_undocumented_members: true
        show_root_heading: true
        show_inherited_members: true
        heading_level: 2
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false


## End of api/evaluation/answer_passage_match.md

---


## File: api/evaluation/SemanticF1.md

# dspy.evaluate.SemanticF1

::: dspy.evaluate.SemanticF1
    handler: python
    options:
        members:
            - __call__
            - batch
            - deepcopy
            - dump_state
            - forward
            - get_lm
            - load
            - load_state
            - map_named_predictors
            - named_parameters
            - named_predictors
            - named_sub_modules
            - parameters
            - predictors
            - reset_copy
            - save
            - set_lm
        show_source: true
        show_undocumented_members: true
        show_root_heading: true
        show_inherited_members: true
        heading_level: 2
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false


## End of api/evaluation/SemanticF1.md

---


## File: api/modules/MultiChainComparison.md

# dspy.MultiChainComparison

::: dspy.MultiChainComparison
    handler: python
    options:
        members:
            - __call__
            - batch
            - deepcopy
            - dump_state
            - forward
            - get_lm
            - load
            - load_state
            - map_named_predictors
            - named_parameters
            - named_predictors
            - named_sub_modules
            - parameters
            - predictors
            - reset_copy
            - save
            - set_lm
        show_source: true
        show_undocumented_members: true
        show_root_heading: true
        show_inherited_members: true
        heading_level: 2
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false


## End of api/modules/MultiChainComparison.md

---


## File: api/modules/Program.md

# dspy.Program

::: dspy.Program
    handler: python
    options:
        members:
            - __call__
            - batch
            - deepcopy
            - dump_state
            - get_lm
            - load
            - load_state
            - map_named_predictors
            - named_parameters
            - named_predictors
            - named_sub_modules
            - parameters
            - predictors
            - reset_copy
            - save
            - set_lm
        show_source: true
        show_undocumented_members: true
        show_root_heading: true
        show_inherited_members: true
        heading_level: 2
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false


## End of api/modules/Program.md

---


## File: api/modules/ProgramOfThought.md

# dspy.ProgramOfThought

::: dspy.ProgramOfThought
    handler: python
    options:
        members:
            - __call__
            - batch
            - deepcopy
            - dump_state
            - forward
            - get_lm
            - load
            - load_state
            - map_named_predictors
            - named_parameters
            - named_predictors
            - named_sub_modules
            - parameters
            - predictors
            - reset_copy
            - save
            - set_lm
        show_source: true
        show_undocumented_members: true
        show_root_heading: true
        show_inherited_members: true
        heading_level: 2
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false


## End of api/modules/ProgramOfThought.md

---


## File: api/modules/Module.md

# dspy.Module

::: dspy.Module
    handler: python
    options:
        members:
            - __call__
            - batch
            - deepcopy
            - dump_state
            - get_lm
            - load
            - load_state
            - map_named_predictors
            - named_parameters
            - named_predictors
            - named_sub_modules
            - parameters
            - predictors
            - reset_copy
            - save
            - set_lm
        show_source: true
        show_undocumented_members: true
        show_root_heading: true
        show_inherited_members: true
        heading_level: 2
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false


## End of api/modules/Module.md

---


## File: api/modules/Parallel.md

# dspy.Parallel

::: dspy.Parallel
    handler: python
    options:
        members:
            - __call__
            - forward
        show_source: true
        show_undocumented_members: true
        show_root_heading: true
        show_inherited_members: true
        heading_level: 2
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false


## End of api/modules/Parallel.md

---


## File: api/modules/ChainOfThought.md

# dspy.ChainOfThought

::: dspy.ChainOfThought
    handler: python
    options:
        members:
            - __call__
            - batch
            - deepcopy
            - dump_state
            - forward
            - get_lm
            - load
            - load_state
            - map_named_predictors
            - named_parameters
            - named_predictors
            - named_sub_modules
            - parameters
            - predictors
            - reset_copy
            - save
            - set_lm
        show_source: true
        show_undocumented_members: true
        show_root_heading: true
        show_inherited_members: true
        heading_level: 2
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false


## End of api/modules/ChainOfThought.md

---


## File: api/modules/ChainOfThoughtWithHint.md

# dspy.ChainOfThoughtWithHint

::: dspy.ChainOfThoughtWithHint
    handler: python
    options:
        members:
            - __call__
            - batch
            - deepcopy
            - dump_state
            - forward
            - get_lm
            - load
            - load_state
            - map_named_predictors
            - named_parameters
            - named_predictors
            - named_sub_modules
            - parameters
            - predictors
            - reset_copy
            - save
            - set_lm
        show_source: true
        show_undocumented_members: true
        show_root_heading: true
        show_inherited_members: true
        heading_level: 2
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false


## End of api/modules/ChainOfThoughtWithHint.md

---


## File: api/modules/ReAct.md

# dspy.ReAct

::: dspy.ReAct
    handler: python
    options:
        members:
            - __call__
            - batch
            - deepcopy
            - dump_state
            - forward
            - get_lm
            - load
            - load_state
            - map_named_predictors
            - named_parameters
            - named_predictors
            - named_sub_modules
            - parameters
            - predictors
            - reset_copy
            - save
            - set_lm
            - truncate_trajectory
        show_source: true
        show_undocumented_members: true
        show_root_heading: true
        show_inherited_members: true
        heading_level: 2
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false


## End of api/modules/ReAct.md

---


## File: api/modules/Refine.md

# dspy.Refine

::: dspy.Refine
    handler: python
    options:
        members:
            - __call__
            - batch
            - deepcopy
            - dump_state
            - forward
            - get_lm
            - load
            - load_state
            - map_named_predictors
            - named_parameters
            - named_predictors
            - named_sub_modules
            - parameters
            - predictors
            - reset_copy
            - save
            - set_lm
        show_source: true
        show_undocumented_members: true
        show_root_heading: true
        show_inherited_members: true
        heading_level: 2
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false


## End of api/modules/Refine.md

---


## File: api/modules/BestOfN.md

# dspy.BestOfN

::: dspy.BestOfN
    handler: python
    options:
        members:
            - __call__
            - batch
            - deepcopy
            - dump_state
            - forward
            - get_lm
            - load
            - load_state
            - map_named_predictors
            - named_parameters
            - named_predictors
            - named_sub_modules
            - parameters
            - predictors
            - reset_copy
            - save
            - set_lm
        show_source: true
        show_undocumented_members: true
        show_root_heading: true
        show_inherited_members: true
        heading_level: 2
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false


## End of api/modules/BestOfN.md

---


## File: api/modules/Predict.md

# dspy.Predict

::: dspy.Predict
    handler: python
    options:
        members:
            - __call__
            - batch
            - deepcopy
            - dump_state
            - forward
            - get_config
            - get_lm
            - load
            - load_state
            - map_named_predictors
            - named_parameters
            - named_predictors
            - named_sub_modules
            - parameters
            - predictors
            - reset
            - reset_copy
            - save
            - set_lm
            - update_config
        show_source: true
        show_undocumented_members: true
        show_root_heading: true
        show_inherited_members: true
        heading_level: 2
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false


## End of api/modules/Predict.md

---


## File: tutorials/index.md

* [Retrieval-Augmented Generation](/tutorials/rag/)

* [Agents](/tutorials/agents/)

* [Reasoning](/tutorials/math/)

* [Entity Extraction](/tutorials/entity_extraction/)

* [Classification](/tutorials/classification/)

* [Classification Finetuning](/tutorials/classification_finetuning/)

* [Multi-Hop Search](/tutorials/multihop_search/)

* [Privacy-Conscious Delegation](/tutorials/papillon/)

* [Image Generation Prompt iteration](/tutorials/image_generation_prompting/)

* [Output Refinement](/tutorials/output_refinement/best-of-n-and-refine/)

* [Finetuning Agents](/tutorials/games/)

* [Saving and Loading](/tutorials/saving/index.md)

* [Deployment](/tutorials/deployment/)

* [Debugging & Observability](/tutorials/observability/)


We are working on upgrading more tutorials and other examples to [DSPy 2.5](https://github.com/stanfordnlp/dspy/blob/main/examples/migration.ipynb) from earlier DSPy versions.

## End of tutorials/index.md

---


## File: tutorials/classification/index.md

Please refer to [this tutorial from Drew Breunig](https://www.dbreunig.com/2024/12/12/pipelines-prompt-optimization-with-dspy.html) using DSPy.

This tutorial demonstrates a few aspects of using DSPy in a highly-accessible, concrete context for categorizing historic events with a tiny LM.

## End of tutorials/classification/index.md

---


## File: tutorials/output_refinement/best-of-n-and-refine.md

# Output Refinement: BestOfN and Refine

Both `BestOfN` and `Refine` are DSPy modules designed to improve the reliability and quality of predictions by making multiple `LM` calls with different parameter settings. Both modules stop when they have reached `N` attempts or when the `reward_fn` returns an award above the `threshold`.

## BestOfN

`BestOfN` is a module that runs the provided module multiple times (up to `N`) with different temperature settings. It returns either the first prediction that passes a specified threshold or the one with the highest reward if none meets the threshold.

### Basic Usage

Lets say we wanted to have the best chance of getting a one word answer from the model. We could use `BestOfN` to try multiple temperature settings and return the best result.

```python
import dspy

def one_word_answer(args, pred: dspy.Prediction) -> float:
    return 1.0 if len(pred.answer.split()) == 1 else 0.0

best_of_3 = dspy.BestOfN(
    module=dspy.ChainOfThought("question -> answer"), 
    N=3, 
    reward_fn=one_word_answer, 
    threshold=1.0
)

result = best_of_3(question="What is the capital of Belgium?")
print(result.answer)  # Brussels
```

### Error Handling

By default, if the module encounters an error during an attempt, it will continue trying until it reaches `N` attempts. You can adjust this behavior with the `fail_count` parameter:

```python
best_of_3 = dspy.BestOfN(
    module=qa, 
    N=3, 
    reward_fn=one_word_answer, 
    threshold=1.0,
    fail_count=1
)

best_of_3(question="What is the capital of Belgium?")
# raises an error after the first failure
```

## Refine

`Refine` extends the functionality of `BestOfN` by adding an automatic feedback loop. After each unsuccessful attempt (except the final one), it automatically generates detailed feedback about the module's performance and uses this feedback as hints for subsequent runs.

### Basic Usage

```python
import dspy

def one_word_answer(args, pred: dspy.Prediction) -> float:
    return 1.0 if len(pred.answer.split()) == 1 else 0.0

refine = dspy.Refine(
    module=dspy.ChainOfThought("question -> answer"), 
    N=3, 
    reward_fn=one_word_answer, 
    threshold=1.0
)

result = refine(question="What is the capital of Belgium?")
print(result.answer)  # Brussels
```

### Error Handling

Like `BestOfN`, `Refine` will try up to `N` times by default, even if errors occur. You can control this with the `fail_count` parameter:

```python
# Stop after the first error
refine = dspy.Refine(
    module=qa, 
    N=3, 
    reward_fn=one_word_answer, 
    threshold=1.0,
    fail_count=1
)
```

## Comparison: BestOfN vs. Refine

Both modules serve similar purposes but differ in their approach:

- `BestOfN` simply tries different temperature settings and selects the best resulting prediction as defined by the `reward_fn`.
- `Refine` adds an feedback loop, using the lm to generate a detailed feedback about the module's own performance using the previous prediction and the code in the `reward_fn`. This feedback is then used as hints for subsequent runs.

## Practical Examples

### Ensuring Factual Correctness

```python
import dspy

class FactualityJudge(dspy.Signature):
    """Determine if a statement is factually accurate."""
    statement: str = dspy.InputField()
    is_factual: bool = dspy.OutputField()

factuality_judge = dspy.ChainOfThought(FactualityJudge)

def factuality_reward(args, pred: dspy.Prediction) -> float:
    statement = pred.answer    
    result = factuality_judge(statement)    
    return 1.0 if result.is_factual else 0.0

refined_qa = dspy.Refine(
    module=dspy.ChainOfThought("question -> answer"),
    N=3,
    reward_fn=factuality_reward,
    threshold=1.0
)

result = refined_qa(question="Tell me about Belgium's capital city.")
print(result.answer)
```

### Summarization - Controlling Response Length

```python
import dspy

def ideal_length_reward(args, pred: dspy.Prediction) -> float:
    """
    Reward the summary for being close to 75 words with a tapering off for longer summaries.
    """
    word_count = len(pred.summary.split())
    distance = abs(word_count - 75)
    return max(0.0, 1.0 - (distance / 125))

optimized_summarizer = dspy.BestOfN(
    module=dspy.ChainOfThought("text -> summary"),
    N=50,
    reward_fn=ideal_length_reward,
    threshold=0.9
)

result = optimized_summarizer(
    text="[Long text to summarize...]"
)
print(result.summary)
```

## Migration from `dspy.Suggest` and `dspy.Assert`

`BestOfN` and `Refine` are the replacements for `dspy.Suggest` and `dspy.Assert` as of DSPy 2.6.


## End of tutorials/output_refinement/best-of-n-and-refine.md

---


## File: tutorials/old/other_tutorial.md

---
sidebar_position: 99999
---

# Additional Resources

## Tutorials

| **Level** |  **Tutorial** |  **Run in Colab** |  **Description** |
| --- | -------------  |  -------------  |  -------------  | 
| Beginner |  [**Getting Started**](https://github.com/stanfordnlp/dspy/blob/main/intro.ipynb) | [<img align="center" src="https://colab.research.google.com/assets/colab-badge.svg" />](https://colab.research.google.com/github/stanfordnlp/dspy/blob/main/intro.ipynb)  |  Introduces the basic building blocks in DSPy. Tackles the task of complex question answering with HotPotQA. |
| Beginner | [**Minimal Working Example**](/docs/docs/quick-start/getting-started-01.md) | N/A | Builds a very simple chain-of-thought program in DSPy for question answering. Very short. |
| Beginner | [**Compiling for Tricky Tasks**](https://github.com/stanfordnlp/dspy/blob/main/examples/nli/scone/scone.ipynb) | N/A | Teaches LMs to reason about logical statements and negation. Uses GPT-4 to bootstrap few-shot CoT demonstrations for GPT-3.5. Establishes a state-of-the-art result on [ScoNe](https://arxiv.org/abs/2305.19426). Contributed by [Chris Potts](https://twitter.com/ChrisGPotts/status/1740033519446057077). |
| Beginner | [**Local Models & Custom Datasets**](https://github.com/stanfordnlp/dspy/blob/main/skycamp2023.ipynb) | [<img align="center" src="https://colab.research.google.com/assets/colab-badge.svg" />](https://colab.research.google.com/github/stanfordnlp/dspy/blob/main/skycamp2023.ipynb) | Illustrates two different things together: how to use local models (Llama-2-13B in particular) and how to use your own data examples for training and development.
| Intermediate | [**The DSPy Paper**](https://arxiv.org/abs/2310.03714) | N/A | Sections 3, 5, 6, and 7 of the DSPy paper can be consumed as a tutorial. They include explained code snippets, results, and discussions of the abstractions and API.
| Intermediate | [**DSPy Assertions**](https://arxiv.org/abs/2312.13382) | [<img align="center" src="https://colab.research.google.com/assets/colab-badge.svg" />](https://colab.research.google.com/github/stanfordnlp/dspy/blob/main/examples/longformqa/longformqa_assertions.ipynb) | Introduces example of applying DSPy Assertions while generating long-form responses to questions with citations. Presents comparative evaluation in both zero-shot and compiled settings.
| Intermediate | [**Finetuning for Complex Programs**](https://twitter.com/lateinteraction/status/1712135660797317577) | [<img align="center" src="https://colab.research.google.com/assets/colab-badge.svg" />](https://colab.research.google.com/github/stanfordnlp/dspy/blob/main/examples/qa/hotpot/multihop_finetune.ipynb) | Teaches a local T5 model (770M) to do exceptionally well on HotPotQA. Uses only 200 labeled answers. Uses no hand-written prompts, no calls to OpenAI, and no labels for retrieval or reasoning.
| Advanced | [**Information Extraction**](https://twitter.com/KarelDoostrlnck/status/1724991014207930696) | [<img align="center" src="https://colab.research.google.com/assets/colab-badge.svg" />](https://colab.research.google.com/drive/1CpsOiLiLYKeGrhmq579_FmtGsD5uZ3Qe) | Tackles extracting information from long articles (biomedical research papers). Combines in-context learning and retrieval to set SOTA on BioDEX. Contributed by [Karel D’Oosterlinck](https://twitter.com/KarelDoostrlnck/status/1724991014207930696).  |


## Resources

- [DSPy talk at ScaleByTheBay Nov 2023](https://www.youtube.com/watch?v=Dt3H2ninoeY).
- [DSPy webinar with MLOps Learners](https://www.youtube.com/watch?v=im7bCLW2aM4), a bit longer with Q&A.
- Hands-on Overviews of DSPy by the community: [DSPy Explained! by Connor Shorten](https://www.youtube.com/watch?v=41EfOY0Ldkc), [DSPy explained by code_your_own_ai](https://www.youtube.com/watch?v=ycfnKPxBMck), [DSPy Crash Course by AI Bites](https://youtu.be/5-zgASQKkKQ?si=3gnmVouT5_rpk_nu)
- Interviews: [Weaviate Podcast in-person](https://www.youtube.com/watch?v=CDung1LnLbY), and you can find 6-7 other remote podcasts on YouTube from a few different perspectives/audiences.
- **Tracing in DSPy** with Arize Phoenix: [Tutorial for tracing your prompts and the steps of your DSPy programs](https://colab.research.google.com/github/Arize-ai/phoenix/blob/main/tutorials/tracing/dspy_tracing_tutorial.ipynb)
- **Tracing & Optimization Tracking in DSPy** with Parea AI: [Tutorial on tracing & evaluating a DSPy RAG program](https://docs.parea.ai/tutorials/dspy-rag-trace-evaluate/tutorial)
- **Prompt Optimization with DSPy and G-Eval Metrics** by Alberto Romero: [Medium article](https://medium.com/@a-romero/prompt-optimization-with-dspy-and-g-eval-metrics-e7d0bdd21b8b), [Repo](https://github.com/a-romero/dspy-risk-assessment), [Video](https://youtu.be/kK30U-XiiNI)

## End of tutorials/old/other_tutorial.md

---


## File: tutorials/old/simplified-baleen.md

---
sidebar_position: 2
---

# [02] Multi-Hop Question Answering

A single search query is often not enough for complex QA tasks. For instance, an example within `HotPotQA` includes a question about the birth city of the writer of "Right Back At It Again". A search query often identifies the author correctly as "Jeremy McKinnon", but lacks the capability to compose the intended answer in determining when he was born.

The standard approach for this challenge in retrieval-augmented NLP literature is to build multi-hop search systems, like GoldEn (Qi et al., 2019) and Baleen (Khattab et al., 2021). These systems read the retrieved results and then generate additional queries to gather additional information when necessary before arriving to a final answer. Using DSPy, we can easily simulate such systems in a few lines of code.

## Configuring LM and RM

We'll start by setting up the language model (LM) and retrieval model (RM), which **DSPy** supports through multiple [LM](/category/language-model-clients) and [RM](/category/retrieval-model-clients) APIs and [local models hosting](/category/local-language-model-clients).

In this notebook, we'll work with GPT-3.5 (`gpt-3.5-turbo`) and the `ColBERTv2` retriever (a free server hosting a Wikipedia 2017 "abstracts" search index containing the first paragraph of each article from this [2017 dump](https://hotpotqa.github.io/wiki-readme.html)). We configure the LM and RM within DSPy, allowing DSPy to internally call the respective module when needed for generation or retrieval. 

```python
import dspy

turbo = dspy.OpenAI(model='gpt-3.5-turbo')
colbertv2_wiki17_abstracts = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')

dspy.settings.configure(lm=turbo, rm=colbertv2_wiki17_abstracts)
```

## Loading the Dataset

For this tutorial, we make use of the mentioned `HotPotQA` dataset, a collection of complex question-answer pairs typically answered in a multi-hop fashion. We can load this dataset provided by DSPy through the `HotPotQA` class:

```python
from dspy.datasets import HotPotQA

# Load the dataset.
dataset = HotPotQA(train_seed=1, train_size=20, eval_seed=2023, dev_size=50, test_size=0)

# Tell DSPy that the 'question' field is the input. Any other fields are labels and/or metadata.
trainset = [x.with_inputs('question') for x in dataset.train]
devset = [x.with_inputs('question') for x in dataset.dev]

len(trainset), len(devset)
```
**Output:**
```text
(20, 50)
```

## Building Signature

Now that we have the data loaded let's start defining the signatures for sub-tasks of out Baleen pipeline.

We'll start by creating the `GenerateAnswer` signature that'll take `context` and `question` as input and give `answer` as output.

```python
class GenerateAnswer(dspy.Signature):
    """Answer questions with short factoid answers."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")
```

Unlike usual QA pipelines, we have an intermediate question-generation step in Baleen for which we'll need to define a new Signature for the "hop" behavior: inputting some context and a question to generate a search query to find missing information. 

```python
class GenerateSearchQuery(dspy.Signature):
    """Write a simple search query that will help answer a complex question."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    query = dspy.OutputField()
```

!!! info
    We could have written `context = GenerateAnswer.signature.context` to avoid duplicating the description of the context field.

Now that we have the necessary signatures in place, we can start building the Baleen pipeline!

## Building the Pipeline

So, let's define the program itself `SimplifiedBaleen`. There are many possible ways to implement this, but we'll keep this version down to the key elements.

```python
from dsp.utils import deduplicate

class SimplifiedBaleen(dspy.Module):
    def __init__(self, passages_per_hop=3, max_hops=2):
        super().__init__()

        self.generate_query = [dspy.ChainOfThought(GenerateSearchQuery) for _ in range(max_hops)]
        self.retrieve = dspy.Retrieve(k=passages_per_hop)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
        self.max_hops = max_hops
    
    def forward(self, question):
        context = []
        
        for hop in range(self.max_hops):
            query = self.generate_query[hop](context=context, question=question).query
            passages = self.retrieve(query).passages
            context = deduplicate(context + passages)

        pred = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=pred.answer)
```

As we can see, the `__init__` method defines a few key sub-modules:

- **generate_query**: For each hop, we will have one `dspy.ChainOfThought` predictor with the `GenerateSearchQuery` signature.
- **retrieve**: This module will conduct the search using the generated queries over our defined ColBERT RM search index via the `dspy.Retrieve` module.
- **generate_answer**: This `dspy.Predict` module will be used with the `GenerateAnswer` signature to produce the final answer.

The `forward` method uses these sub-modules in simple control flow.

1. First, we'll loop up to `self.max_hops` times.
2. In each iteration, we'll generate a search query using the predictor at `self.generate_query[hop]`.
3. We'll retrieve the top-k passages using that query.
4. We'll add the (deduplicated) passages to our `context` accumulator.
5. After the loop, we'll use `self.generate_answer` to produce an answer.
6. We'll return a prediction with the retrieved `context` and predicted `answer`.

## Executing the Pipeline

Let's execute this program in its zero-shot (uncompiled) setting. 

This doesn't necessarily imply the performance will be bad but rather that we're bottlenecked directly by the reliability of the underlying LM to understand our sub-tasks from minimal instructions. Often, this is perfectly fine when using the most expensive/powerful models (e.g., GPT-4) on the easiest and most standard tasks (e.g., answering simple questions about popular entities).

```python
# Ask any question you like to this simple RAG program.
my_question = "How many storeys are in the castle that David Gregory inherited?"

# Get the prediction. This contains `pred.context` and `pred.answer`.
uncompiled_baleen = SimplifiedBaleen()  # uncompiled (i.e., zero-shot) program
pred = uncompiled_baleen(my_question)

# Print the contexts and the answer.
print(f"Question: {my_question}")
print(f"Predicted Answer: {pred.answer}")
print(f"Retrieved Contexts (truncated): {[c[:200] + '...' for c in pred.context]}")
```
**Output:**
```text
Question: How many storeys are in the castle that David Gregory inherited?
Predicted Answer: five
Retrieved Contexts (truncated): ['David Gregory (physician) | David Gregory (20 December 1625 – 1720) was a Scottish physician and inventor. His surname is sometimes spelt as Gregorie, the original Scottish spelling. He inherited Kinn...', 'The Boleyn Inheritance | The Boleyn Inheritance is a novel by British author Philippa Gregory which was first published in 2006. It is a direct sequel to her previous novel "The Other Boleyn Girl," an...', 'Gregory of Gaeta | Gregory was the Duke of Gaeta from 963 until his death. He was the second son of Docibilis II of Gaeta and his wife Orania. He succeeded his brother John II, who had left only daugh...', 'Kinnairdy Castle | Kinnairdy Castle is a tower house, having five storeys and a garret, two miles south of Aberchirder, Aberdeenshire, Scotland. The alternative name is Old Kinnairdy....', 'Kinnaird Head | Kinnaird Head (Scottish Gaelic: "An Ceann Àrd" , "high headland") is a headland projecting into the North Sea, within the town of Fraserburgh, Aberdeenshire on the east coast of Scotla...', 'Kinnaird Castle, Brechin | Kinnaird Castle is a 15th-century castle in Angus, Scotland. The castle has been home to the Carnegie family, the Earl of Southesk, for more than 600 years....']
```

We can inspect the last **three** calls to the LM (i.e., generating the first hop's query, generating the second hop's query, and generating the answer) using:

```python
turbo.inspect_history(n=3)
```

## Optimizing the Pipeline

However, a zero-shot approach quickly falls short for more specialized tasks, novel domains/settings, and more efficient (or open) models. 

To address this, **DSPy** offers compilation. Let's compile our multi-hop (`SimplifiedBaleen`) program. 

Let's first define our validation logic for compilation: 

- The predicted answer matches the gold answer.
- The retrieved context contains the gold answer.
- None of the generated queries is rambling (i.e., none exceeds 100 characters in length).
- None of the generated queries is roughly repeated (i.e., none is within 0.8 or higher F1 score of earlier queries).

```python
def validate_context_and_answer_and_hops(example, pred, trace=None):
    if not dspy.evaluate.answer_exact_match(example, pred): return False
    if not dspy.evaluate.answer_passage_match(example, pred): return False

    hops = [example.question] + [outputs.query for *_, outputs in trace if 'query' in outputs]

    if max([len(h) for h in hops]) > 100: return False
    if any(dspy.evaluate.answer_exact_match_str(hops[idx], hops[:idx], frac=0.8) for idx in range(2, len(hops))): return False

    return True
```

We'll use one of the most basic teleprompters in **DSPy**, namely, `BootstrapFewShot` to optimize the predictors in pipeline with few-shot examples.

```python
from dspy.teleprompt import BootstrapFewShot

teleprompter = BootstrapFewShot(metric=validate_context_and_answer_and_hops)
compiled_baleen = teleprompter.compile(SimplifiedBaleen(), teacher=SimplifiedBaleen(passages_per_hop=2), trainset=trainset)
```

## Evaluating the Pipeline

Let's now define our evaluation function and compare the performance of the uncompiled and compiled Baleen pipelines. While this devset does not serve as a completely reliable benchmark, it is instructive to use for this tutorial. 

```python
from dspy.evaluate.evaluate import Evaluate

# Define metric to check if we retrieved the correct documents
def gold_passages_retrieved(example, pred, trace=None):
    gold_titles = set(map(dspy.evaluate.normalize_text, example["gold_titles"]))
    found_titles = set(
        map(dspy.evaluate.normalize_text, [c.split(" | ")[0] for c in pred.context])
    )
    return gold_titles.issubset(found_titles)

# Set up the `evaluate_on_hotpotqa` function. We'll use this many times below.
evaluate_on_hotpotqa = Evaluate(devset=devset, num_threads=1, display_progress=True, display_table=5)

uncompiled_baleen_retrieval_score = evaluate_on_hotpotqa(uncompiled_baleen, metric=gold_passages_retrieved, display=False)

compiled_baleen_retrieval_score = evaluate_on_hotpotqa(compiled_baleen, metric=gold_passages_retrieved)

print(f"## Retrieval Score for uncompiled Baleen: {uncompiled_baleen_retrieval_score}")
print(f"## Retrieval Score for compiled Baleen: {compiled_baleen_retrieval_score}")
```
**Output:**
```text
## Retrieval Score for uncompiled Baleen: 36.0
## Retrieval Score for compiled Baleen: 60.0
```

Excellent! There might be something to this compiled, multi-hop program then.

Earlier, we said simple programs are not very effective at finding all evidence required for answering each question. Is this resolved by the adding some greater prompting techniques in the forward function of `SimplifiedBaleen`? Does compiling programs improve performance?

While in our tutorial we demonstrate our findings, the answer for these questions will not always be obvious. However, DSPy makes it extremely easy to try out the many diverse approaches with minimal effort. 

Now that you've seen a example of how to build a simple yet powerful pipeline, it's time for you to build one yourself!


## End of tutorials/old/simplified-baleen.md

---


## File: tutorials/old/examples.md

---
sidebar_position: 99998
---

# Community Examples

The DSPy team believes complexity has to be justified. We take this seriously: we never release a complex tutorial (above) or example (below) _unless we can demonstrate empirically that this complexity has generally led to improved quality or cost._ This kind of rule is rarely enforced by other frameworks or docs, but you can count on it in DSPy examples.

There's a bunch of examples in the `examples/` directory and in the top-level directory. We welcome contributions!

You can find other examples tweeted by [@lateinteraction](https://twitter.com/lateinteraction) on Twitter/X.

**Some other examples (not exhaustive, feel free to add more via PR):**

- Applying DSPy Assertions
  - [Long-form Answer Generation with Citations, by Arnav Singhvi](https://colab.research.google.com/github/stanfordnlp/dspy/blob/main/examples/longformqa/longformqa_assertions.ipynb)
  - [Generating Answer Choices for Quiz Questions, by Arnav Singhvi](https://colab.research.google.com/github/stanfordnlp/dspy/blob/main/examples/quiz/quiz_assertions.ipynb)
  - [Generating Tweets for QA, by Arnav Singhvi](https://colab.research.google.com/github/stanfordnlp/dspy/blob/main/examples/tweets/tweets_assertions.ipynb)
- [Compiling LCEL runnables from LangChain in DSPy](https://github.com/stanfordnlp/dspy/blob/main/examples/tweets/compiling_langchain.ipynb)
- [AI feedback, or writing LM-based metrics in DSPy](https://github.com/stanfordnlp/dspy/blob/main/examples/tweets/tweet_metric.py)
- [DSPy Optimizers Benchmark on a bunch of different tasks, by Michael Ryan](https://github.com/stanfordnlp/dspy/tree/main/testing/tasks)
- [Indian Languages NLI with gains due to compiling by Saiful Haq](https://github.com/saifulhaq95/DSPy-Indic/blob/main/indicxlni.ipynb)
- [Sophisticated Extreme Multi-Class Classification, IReRa, by Karel D’Oosterlinck](https://github.com/KarelDO/xmc.dspy)
- [DSPy on BIG-Bench Hard Example, by Chris Levy](https://drchrislevy.github.io/posts/dspy/dspy.html)
- [Using Ollama with DSPy for Mistral (quantized) by @jrknox1977](https://gist.github.com/jrknox1977/78c17e492b5a75ee5bbaf9673aee4641)
- [Using DSPy, "The Unreasonable Effectiveness of Eccentric Automatic Prompts" (paper) by VMware's Rick Battle & Teja Gollapudi, and interview at TheRegister](https://www.theregister.com/2024/02/22/prompt_engineering_ai_models/)
- [Portkey with DSPy: Efficiently Track Metrics, Optimize Performance, and Scale Across 250+ LLMs](https://docs.portkey.ai/docs/integrations/libraries/dspy)
- [Langtrace AI DSPy Examples Repository](https://github.com/Scale3-Labs/dspy-examples)
- [Langtrace Blogpost - Build a summarization system using DSPy](https://www.langtrace.ai/blog/build-a-reliable-summarization-system-using-dspy-and-langtrace)

There are also recent cool examples at [Weaviate's DSPy cookbook](https://github.com/weaviate/recipes/tree/main/integrations/llm-frameworks/dspy) by Connor Shorten. [See tutorial on YouTube](https://www.youtube.com/watch?v=CEuUG4Umfxs).


## End of tutorials/old/examples.md

---


## File: tutorials/old/rag.md

---
sidebar_position: 1
---

# [01] RAG: Retrieval-Augmented Generation

Retrieval-augmented generation (RAG) is an approach that allows LLMs to tap into a large corpus of knowledge from sources and query its knowledge store to find relevant passages/content and produce a well-refined response.

RAG ensures LLMs can dynamically utilize real-time knowledge even if not originally trained on the subject and give thoughtful answers. However, with this nuance comes greater complexities in setting up refined RAG pipelines. To reduce these intricacies, we turn to **DSPy**, which offers a seamless approach to setting up prompting pipelines!

## Configuring LM and RM

We'll start by setting up the language model (LM) and retrieval model (RM), which **DSPy** supports through multiple [LM](/building-blocks/1-language_models.md) and [RM](/deep-dive/retrieval_models_clients/Azure.md) APIs and [local models hosting](/deep-dive/language_model_clients/local_models/HFClientTGI).

In this notebook, we'll work with GPT-3.5 (`gpt-3.5-turbo`) and the `ColBERTv2` retriever (a free server hosting a Wikipedia 2017 "abstracts" search index containing the first paragraph of each article from this [2017 dump](https://hotpotqa.github.io/wiki-readme.html)). We configure the LM and RM within DSPy, allowing DSPy to internally call the respective module when needed for generation or retrieval. 

```python
import dspy

turbo = dspy.OpenAI(model='gpt-3.5-turbo')
colbertv2_wiki17_abstracts = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')

dspy.settings.configure(lm=turbo, rm=colbertv2_wiki17_abstracts)
```


## Loading the Dataset

For this tutorial, we make use of the `HotPotQA` dataset, a collection of complex question-answer pairs typically answered in a multi-hop fashion. We can load this dataset provided by DSPy through the `HotPotQA` class:

```python
from dspy.datasets import HotPotQA

# Load the dataset.
dataset = HotPotQA(train_seed=1, train_size=20, eval_seed=2023, dev_size=50, test_size=0)

# Tell DSPy that the 'question' field is the input. Any other fields are labels and/or metadata.
trainset = [x.with_inputs('question') for x in dataset.train]
devset = [x.with_inputs('question') for x in dataset.dev]

len(trainset), len(devset)
```
**Output:**
```text
(20, 50)
```

## Building Signatures

Now that we have the data loaded, let's start defining the [signatures](/building-blocks/2-signatures) for the sub-tasks of our pipeline.

We can identify our simple input `question` and output `answer`, but since we are building out a RAG pipeline, we wish to utilize some contextual information from our ColBERT corpus. So let's define our signature: `context, question --> answer`.

```python
class GenerateAnswer(dspy.Signature):
    """Answer questions with short factoid answers."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")
```

We include small descriptions for the `context` and `answer` fields to define more robust guidelines on what the model will receive and should generate. 

## Building the Pipeline

We will build our RAG pipeline as a [DSPy module](/building-blocks/3-modules) which will require two methods:

* The `__init__` method will simply declare the sub-modules it needs: `dspy.Retrieve` and `dspy.ChainOfThought`. The latter is defined to implement our `GenerateAnswer` signature.
* The `forward` method will describe the control flow of answering the question using the modules we have: Given a question, we'll search for the top-3 relevant passages and then feed them as context for answer generation.


```python
class RAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()

        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
    
    def forward(self, question):
        context = self.retrieve(question).passages
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)
```

## Optimizing the Pipeline

##### Compiling the RAG program

Having defined this program, let's now **compile** it. [Compiling a program](/building-blocks/6-optimizers) will update the parameters stored in each module. In our setting, this is primarily in the form of collecting and selecting good demonstrations for inclusion within the prompt(s).

Compiling depends on three things:

1. **A training set.** We'll just use our 20 question–answer examples from `trainset` above.
1. **A metric for validation.** We'll define a simple `validate_context_and_answer` that checks that the predicted answer is correct and that the retrieved context actually contains the answer.
1. **A specific teleprompter.** The **DSPy** compiler includes a number of **teleprompters** that can optimize your programs.

```python
from dspy.teleprompt import BootstrapFewShot

# Validation logic: check that the predicted answer is correct.
# Also check that the retrieved context does actually contain that answer.
def validate_context_and_answer(example, pred, trace=None):
    answer_EM = dspy.evaluate.answer_exact_match(example, pred)
    answer_PM = dspy.evaluate.answer_passage_match(example, pred)
    return answer_EM and answer_PM

# Set up a basic teleprompter, which will compile our RAG program.
teleprompter = BootstrapFewShot(metric=validate_context_and_answer)

# Compile!
compiled_rag = teleprompter.compile(RAG(), trainset=trainset)
```

!!! info
    **Teleprompters:** Teleprompters are powerful optimizers that can take any program and learn to bootstrap and select effective prompts for its modules. Hence the name which means "prompting at a distance".

    Different teleprompters offer various tradeoffs in terms of how much they optimize cost versus quality, etc. We will used a simple default `BootstrapFewShot` in the example above.

    _If you're into analogies, you could think of this as your training data, your loss function, and your optimizer in a standard DNN supervised learning setup. Whereas SGD is a basic optimizer, there are more sophisticated (and more expensive!) ones like Adam or RMSProp._

## Executing the Pipeline

Now that we've compiled our RAG program, let's try it out.

```python
# Ask any question you like to this simple RAG program.
my_question = "What castle did David Gregory inherit?"

# Get the prediction. This contains `pred.context` and `pred.answer`.
pred = compiled_rag(my_question)

# Print the contexts and the answer.
print(f"Question: {my_question}")
print(f"Predicted Answer: {pred.answer}")
print(f"Retrieved Contexts (truncated): {[c[:200] + '...' for c in pred.context]}")
```

Excellent. How about we inspect the last prompt for the LM?

```python
turbo.inspect_history(n=1)
```
**Output:**
```text
Answer questions with short factoid answers.

---

Question: At My Window was released by which American singer-songwriter?
Answer: John Townes Van Zandt

Question: "Everything Has Changed" is a song from an album released under which record label ?
Answer: Big Machine Records
...(truncated)
```

Even though we haven't written any of this detailed demonstrations, we see that DSPy was able to bootstrap this 3,000 token prompt for 3-shot retrieval-augmented generation with hard negative passages and uses Chain-of-Thought reasoning within an extremely simply-written program.

This illustrates the power of composition and learning. Of course, this was just generated by a particular teleprompter, which may or may not be perfect in each setting. As you'll see in DSPy, there is a large but systematic space of options you have to optimize and validate with respect to your program's quality and cost.

You can also easily inspect the learned objects themselves.

```python
for name, parameter in compiled_rag.named_predictors():
    print(name)
    print(parameter.demos[0])
    print()
```

## Evaluating the Pipeline

We can now evaluate our `compiled_rag` program on the dev set. Of course, this tiny set is _not_ meant to be a reliable benchmark, but it'll be instructive to use it for illustration.

Let's evaluate the accuracy (exact match) of the predicted answer.

```python
from dspy.evaluate.evaluate import Evaluate

# Set up the `evaluate_on_hotpotqa` function. We'll use this many times below.
evaluate_on_hotpotqa = Evaluate(devset=devset, num_threads=1, display_progress=False, display_table=5)

# Evaluate the `compiled_rag` program with the `answer_exact_match` metric.
metric = dspy.evaluate.answer_exact_match
evaluate_on_hotpotqa(compiled_rag, metric=metric)
```
**Output:**
```text
Average Metric: 22 / 50  (44.0): 100%|██████████| 50/50 [00:00<00:00, 116.45it/s]
Average Metric: 22 / 50  (44.0%)

44.0
```

## Evaluating the Retrieval

It may also be instructive to look at the accuracy of retrieval. While there are multiple ways to do this, we can simply check whether the retrieved passages contain the answer.

We can make use of our dev set which includes the gold titles that should be retrieved.

```python
def gold_passages_retrieved(example, pred, trace=None):
    gold_titles = set(map(dspy.evaluate.normalize_text, example['gold_titles']))
    found_titles = set(map(dspy.evaluate.normalize_text, [c.split(' | ')[0] for c in pred.context]))

    return gold_titles.issubset(found_titles)

compiled_rag_retrieval_score = evaluate_on_hotpotqa(compiled_rag, metric=gold_passages_retrieved)
```
**Output:**
```text
Average Metric: 13 / 50  (26.0): 100%|██████████| 50/50 [00:00<00:00, 671.76it/s]Average Metric: 13 / 50  (26.0%)
```

Although this simple `compiled_rag` program is able to answer a decent fraction of the questions correctly (on this tiny set, over 40%), the quality of retrieval is much lower.

This potentially suggests that the LM is often relying on the knowledge it memorized during training to answer questions. To address this weak retrieval, let's explore a second program that involves more advanced search behavior.


## End of tutorials/old/rag.md

---


## File: tutorials/old/summarization.md

---
sidebar_position: 3
---

# [03] Summarization

Summarization is a fundamental task in natural language processing that involves condensing a longer piece of text into a shorter version while retaining its key information and main ideas. It's a crucial skill for both humans and machines, with applications ranging from creating article abstracts to generating concise reports from lengthy documents.

While summarization is valuable, evaluating the quality of summaries produced by language models presents significant challenges. Unlike tasks with clear right or wrong answers, summarization quality is often subjective and context-dependent. Some key difficulties in evaluating summarized outputs include:

1. Balancing information retention with conciseness
2. Preserving the original text's tone and intent
3. Ensuring factual accuracy without introducing errors
4. Adapting to different types of source material and summary purposes
5. Accounting for varying reader preferences and background knowledge

These challenges make it difficult to create universal metrics for summary quality, often requiring a combination of automated measures and human judgment for comprehensive evaluation.

In this example, we show how you can build a summarization system using DSPy. The technique involves the following:

1. We build a DSPy program for doing the actual summarization task which takes a `passage` as input and gives a `summary` as output.
2. We would ideally want to optimize the summarization program over a metric in order to build an effective program. But since writing a metric function for summarization tasks is subjective and ambiguous in nature, we can use a LM to do this task which in turn means we can write another DSPy program to do the task of scoring a summary based on the variable we care about.
3. Finally, we compose the system by using the Metric program for defining the metric function in the Summarization program.

## Metric program

Define two signatures for scoring the summary.

1. The first signature breaks down the summary and assigns defined labels as grades.
```
import dspy

class Breakdown(dspy.Signature):
    """
    Given a passage, break down the passage into key ideas.
    Enumerate every key idea in the passage and
    assign it an importance grade
    (High, Medium, or Low).
    """

    passage = dspy.InputField()
    key_ideas: str = dspy.OutputField(
        desc="numbered list of one key idea per line,"
        "followed by its importance grade, "
             "e.g. 1. <Idea here>. High.")
    importance_grades: list[str] = dspy.OutputField(
        desc='list of importance grades, '
             'e.g. ["High", "Medium", "Low"].')
```

2. The second signature assigns a binary score using the summary and the break down from the previous signature as inputs
```
class SummaryCorrectness(dspy.Signature):
    """
    Compare a system generated summary to the key ideas in the passage.
    For every key idea supplied,
    assign a binary score based on whether the summary contains it.
    And compute an overall score based on the binary scores.
    """

    key_ideas: str = dspy.InputField(
        desc="key ideas in the passage "
             "for evaluating the summary")
    summary: str = dspy.InputField()
    binary_scores: list[bool] = dspy.OutputField(
        desc="list of binary scores for each key idea, "
             "e.g. [True, False, True]")
    overall_score: float = dspy.OutputField(
        desc="overall score for the summary out of 1.0")
```

3. Now write the actual program that computes a weighted score between 0 to 1 and falls back to the overall_score that's loosely scored by the model

```
class Metric(dspy.Module):
    """
    Compute a score for the correctness of a summary.
    """

    def __init__(self):
        self.breakdown = dspy.ChainOfThought(Breakdown)
        self.assess = dspy.ChainOfThought(SummaryCorrectness)

    def forward(self, example, pred, trace=None):
        breakdown = self.breakdown(
            passage=example.passage
        )
        key_ideas = breakdown.key_ideas
        importance_grades = breakdown.importance_grades

        scores = self.assess(
            key_ideas=key_ideas,
            summary=pred.summary,
        )

        try:
            weight_map = {'High': 1.0, 'Medium': 0.7}
            score = sum(
                weight_map.get(g, 0.2) * int(b)
                for g, b in zip(importance_grades, scores.binary_scores)
            )
            score /= sum(weight_map.get(g, 0.2) for g in importance_grades)

        # pylint: disable=broad-except
        except Exception:
            score = float(scores.overall_score)

        return score if trace is None else score >= 0.75
```

## Summarization Program

First define it's signature

```
class SummarizeSignature(dspy.Signature):
    """
    Given a passage, generate a summary.
    """

    passage = dspy.InputField(desc="a passage to summarize")
    summary: str = dspy.OutputField(
        desc="a concise summary of the passage")
```

Next, define the program

```
class Summarize(dspy.Module):
    def __init__(self):
        self.summarize = dspy.ChainOfThought(SummarizeSignature)

    def forward(self, passage: str):
        summary = self.summarize(
            passage=passage
        )
        return summary
```

Let's jump into the evaluation process now. First, define the language model.

```
lm = dspy.LM("openai/gpt-4o-mini", max_tokens=250)
dspy.settings.configure(lm=lm)
```

Load a dataset. The dataset is a `jsonl` file where each line has a passage, summary and a score.

```
dataset = []
with open('src/summarization/programs/summarize/dataset.jsonl', 'r',
          encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)

        passage = data.get("passage", "")
        summary = data.get("summary", "")
        score = data.get("score", 0)

        example = dspy.Example(passage=passage, summary=summary, score=score)
        dataset.append(example)

trainset = [x.with_inputs("passage") for x in dataset]
```

Now, define the metric. This is where we use the `Metric` program to compute the score for the summary generated by the `Summarization` program.

```
def metric(gold, pred, trace=None):
    metric_program = Metric()
    examp = dspy.Example(passage=gold.passage)
    predicted = dspy.Example(summary=pred)
    pred_score = metric_program(example=examp, pred=predicted)
    gold_score = gold.score
    # check if they are almost equal
    return abs(float(gold_score) - float(pred_score)) < 0.2
```

Create the program and evaluate it.
```
program = Summarize()

evaluate = dspy.Evaluate(devset=trainset, metric=metric,
                    display_progress=True,
                    display_table=True, provide_traceback=True)

res = evaluate(program, devset=trainset)
print(res)
```

That's it! Hopefully this example was helpful in understanding how you can compose DSPy programs to build a truly compound AI system that's optimized for isolated tasks.

## End of tutorials/old/summarization.md

---


## File: tutorials/old/intro-tutorialX.md

<!-- ---
sidebar_position: 1
hide:
  - navigation
  # - toc
---

::: notebook
intro-tutorial.ipynb
::: -->


## End of tutorials/old/intro-tutorialX.md

---


## File: tutorials/observability/index.md

# Tutorial: Debugging and Observability in DSPy

This guide demonstrates how to debug problems and improve observability in DSPy. Modern AI programs often involve multiple components, such as language models, retrievers, and tools. DSPy allows you to build and optimize such complex AI systems in a clean and modular way.

However, as systems grow more sophisticated, the ability to **understand what your system is doing** becomes critical. Without transparency, the prediction process can easily become a black box, making failures or quality issues difficult to diagnose and production maintenance challenging.

By the end of this tutorial, you'll understand how to debug an issue and improve observability using [MLflow Tracing](#tracing). You'll also explore how to build a custom logging solution using callbacks.



## Define a Program

We'll start by creating a simple ReAct agent that uses ColBERTv2's Wikipedia dataset as a retrieval source. You can replace this with a more sophisticated program.

```python
import dspy
from dspy.datasets import HotPotQA

lm = dspy.LM('openai/gpt-4o-mini')
colbert = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')
dspy.configure(lm=lm, rm=colbert)

agent = dspy.ReAct("question -> answer", tools=[dspy.Retrieve(k=1)])
```

Now, let's ask the agent a simple question:

```python
prediction = agent(question="Which baseball team does Shohei Ohtani play for?")
print(prediction.answer)
```

```
Shohei Ohtani plays for the Los Angeles Angels.
```

Oh, this is incorrect. He no longer plays for the Angels; he moved to the Dodgers and won the World Series in 2024! Let's debug the program and explore potential fixes.

## Using ``inspect_history``

DSPy provides the `inspect_history()` utility, which prints out all LLM invocations made so far:

```python
# Print out 5 LLM calls
dspy.inspect_history(n=5)
```

```
[2024-12-01T10:23:29.144257]

System message:

Your input fields are:
1. `question` (str)

...

Response:

[[ ## Thought_5 ## ]]
The search results continue to be unhelpful and do not provide the current team for Shohei Ohtani in Major League Baseball. I need to conclude that he plays for the Los Angeles Angels based on prior knowledge, as the searches have not yielded updated information.

[[ ## Action_5 ## ]]
Finish[Los Angeles Angels] 

[[ ## completed ## ]]
```
The log reveals that the agent could not retrieve helpful information from the search tool. However, what exactly did the retriever return? While useful, `inspect_history` has some limitations:

* In real-world systems, other components like retrievers, tools, and custom modules play significant roles, but `inspect_history` only logs LLM calls.
* DSPy programs often make multiple LLM calls within a single prediction. Monolith log history makes it hard to organize logs, especially when handling multiple questions.
* Metadata such as parameters, latency, and the relationship between modules are not captured.

**Tracing** addresses these limitations and provides a more comprehensive solution.

## Tracing

[MLflow](https://mlflow.org/docs/latest/llms/tracing/index.html) is an end-to-end machine learning platform that is integrated seamlessly with DSPy to support best practices in LLMOps. Using MLflow's automatic tracing capability with DSPy is straightforward; **No sign up for services or an API key is required**. You just need to install MLflow and call `mlflow.dspy.autolog()` in your notebook or script.

```bash
pip install -U mlflow>=2.18.0
```

```python
import mlflow

mlflow.dspy.autolog()

# This is optional. Create an MLflow Experiment to store and organize your traces.
mlflow.set_experiment("DSPy")
```

Now you're all set! Let's run your agent again:

```python
agent(question="Which baseball team does Shohei Ohtani play for?")
```

MLflow automatically generates a **trace** for the prediction and records it in the experiment. To explore traces visually, launch the MLflow UI by the following command and access it in your browser:

```bash
mlflow ui --port 5000
```

![DSPy MLflow Tracing](./dspy-tracing.gif)

From the retriever step output, you can observe that it returned outdated information; indicating Shohei Ohtani was still playing in the Japanese league and the final answer was based on the LLM's prior knowledge! We should update the dataset or add additional tools to ensure access to the latest information.

!!! info Learn more about MLflow

    MLflow is an end-to-end LLMOps platform that offers extensive features like experiment tracking, evaluation, and deployment. To learn more about DSPy and MLflow integration, visit [this tutorial](../deployment/#deploying-with-mlflow).

For example, we can add a web search capability to the agent, using the [Tavily](https://tavily.com/) web search API.

```python
from dspy.predict.react import Tool
from tavily import TavilyClient

search_client = TavilyClient(api_key="<YOUR_TAVILY_API_KEY>")

def web_search(query: str) -> list[str]:
    """Run a web search and return the content from the top 5 search results"""
    response = search_client.search(query)
    return [r["content"] for r in response["results"]]

agent = dspy.ReAct("question -> answer", tools=[Tool(web_search)])

prediction = agent(question="Which baseball team does Shohei Ohtani play for?")
print(agent.answer)
```

```
Los Angeles Dodgers
```


## Building a Custom Logging Solution

Sometimes, you may want to implement a custom logging solution. For instance, you might need to log specific events triggered by a particular module. DSPy's **callback** mechanism supports such use cases. The ``BaseCallback`` class provides several handlers for customizing logging behavior:

|Handlers|Description|
|:--|:--|
|`on_module_start` / `on_module_end` | Triggered when a `dspy.Module` subclass is invoked. |
|`on_lm_start` / `on_lm_end` | Triggered when a `dspy.LM` subclass is invoked. |
|`on_adapter_format_start` / `on_adapter_format_end`| Triggered when a `dspy.Adapter` subclass formats the input prompt. |
|`on_adapter_parse_start` / `on_adapter_parse_end`| Triggered when a `dspy.Adapter` subclass postprocess the output text from an LM. |

Here’s an example of custom callback that logs the intermediate steps of a ReAct agent:

```python
import dspy
from dspy.utils.callback import BaseCallback

# 1. Define a custom callback class that extends BaseCallback class
class AgentLoggingCallback(BaseCallback):

    # 2. Implement on_module_end handler to run a custom logging code.
    def on_module_end(self, call_id, outputs, exception):
        step = "Reasoning" if self._is_reasoning_output(outputs) else "Acting"
        print(f"== {step} Step ===")
        for k, v in outputs.items():
            print(f"  {k}: {v}")
        print("\n")

    def _is_reasoning_output(self, outputs):
        return any(k.startswith("Thought") for k in outputs.keys())

# 3. Set the callback to DSPy setting so it will be applied to program execution
dspy.configure(callbacks=[AgentLoggingCallback()])
```


```
== Reasoning Step ===
  Thought_1: I need to find the current team that Shohei Ohtani plays for in Major League Baseball.
  Action_1: Search[Shohei Ohtani current team 2023]

== Acting Step ===
  passages: ["Shohei Ohtani ..."]

...
```

!!! info Handling Inputs and Outputs in Callbacks

    Be cautious when working with input or output data in callbacks. Mutating them in-place can modify the original data passed to the program, potentially leading to unexpected behavior. To avoid this, it’s strongly recommended to create a copy of the data before performing any operations that may alter it.


## End of tutorials/observability/index.md

---


## File: tutorials/papillon/index.md

Please refer to [this tutorial from the PAPILLON authors](https://github.com/Columbia-NLP-Lab/PAPILLON/blob/main/papillon_tutorial.ipynb) using DSPy.

This tutorial demonstrates a few aspects of using DSPy in a more advanced context:

1. It builds a multi-stage `dspy.Module` that involves a small local LM using an external tool.
2. It builds a multi-stage _judge_ in DSPy, and uses it as a metric for evaluation.
3. It uses this judge for optimizing the `dspy.Module`, using a large model as a teacher for a small local LM.

## End of tutorials/papillon/index.md

---


## File: tutorials/deployment/index.md

# Tutorial: Deploying your DSPy program

This guide demonstrates two potential ways to deploy your DSPy program in production: FastAPI for lightweight deployments and MLflow for more production-grade deployments with program versioning and management.

Below, we'll assume you have the following simple DSPy program that you want to deploy. You can replace this with something more sophisticated.

```python
import dspy

dspy.settings.configure(lm=dspy.LM("openai/gpt-4o-mini"))
dspy_program = dspy.ChainOfThought("question -> answer")
```

## Deploying with FastAPI

FastAPI offers a straightforward way to serve your DSPy program as a REST API. This is ideal when you have direct access to your program code and need a lightweight deployment solution.

```bash
> pip install fastapi uvicorn
> export OPENAI_API_KEY="your-openai-api-key"
```

Let's create a FastAPI application to serve your `dspy_program` defined above.

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import dspy

app = FastAPI(
    title="DSPy Program API",
    description="A simple API serving a DSPy Chain of Thought program",
    version="1.0.0"
)

# Define request model for better documentation and validation
class Question(BaseModel):
    text: str

# Configure your language model and 'asyncify' your DSPy program.
lm = dspy.LM("openai/gpt-4o-mini")
dspy.settings.configure(lm=lm, async_max_workers=4) # default is 8
dspy_program = dspy.ChainOfThought("question -> answer")
dspy_program = dspy.asyncify(dspy_program)

@app.post("/predict")
async def predict(question: Question):
    try:
        result = await dspy_program(question=question.text)
        return {
            "status": "success",
            "data": result.toDict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

In the code above, we call `dspy.asyncify` to convert the dspy program to run in async mode for high-throughput FastAPI
deployments. Currently, this runs the dspy program in a separate thread and awaits its result.

By default, the limit of spawned threads is 8. Think of this like a worker pool.
If you have 8 in-flight programs and call it once more, the 9th call will wait until one of the 8 returns.
You can configure the async capacity using the new `async_max_workers` setting.

??? "Streaming, in DSPy 2.6.0+"

    Streaming is also supported in DSPy 2.6.0+, available as a release candidate via `pip install -U --pre dspy`.

    We can use `dspy.streamify` to convert the dspy program to a streaming mode. This is useful when you want to stream
    the intermediate outputs (i.e. O1-style reasoning) to the client before the final prediction is ready. This uses
    asyncify under the hood and inherits the execution semantics.

    ```python
    dspy_program = dspy.asyncify(dspy.ChainOfThought("question -> answer"))
    streaming_dspy_program = dspy.streamify(dspy_program)

    @app.post("/predict/stream")
    async def stream(question: Question):
        async def generate():
            async for value in streaming_dspy_program(question=question.text):
                if isinstance(value, dspy.Prediction):
                    data = {"prediction": value.labels().toDict()}
                elif isinstance(value, litellm.ModelResponse):
                    data = {"chunk": value.json()}
                yield f"data: {ujson.dumps(data)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    # Since you're often going to want to stream the result of a DSPy program as server-sent events,
    # we've included a helper function for that, which is equivalent to the code above.

    from dspy.utils.streaming import streaming_response

    @app.post("/predict/stream")
    async def stream(question: Question):
        stream = streaming_dspy_program(question=question.text)
        return StreamingResponse(streaming_response(stream), media_type="text/event-stream")
    ```

Write your code to a file, e.g., `fastapi_dspy.py`. Then you can serve the app with:

```bash
> uvicorn fastapi_dspy:app --reload
```

It will start a local server at `http://127.0.0.1:8000/`. You can test it with the python code below:

```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={"text": "What is the capital of France?"}
)
print(response.json())
```

You should see the response like below:

```json
{'status': 'success', 'data': {'reasoning': 'The capital of France is a well-known fact, commonly taught in geography classes and referenced in various contexts. Paris is recognized globally as the capital city, serving as the political, cultural, and economic center of the country.', 'answer': 'The capital of France is Paris.'}}
```

## Deploying with MLflow

We recommend deploying with MLflow if you are looking to package your DSPy program and deploy in an isolated environment.
MLflow is a popular platform for managing machine learning workflows, including versioning, tracking, and deployment.

```bash
> pip install mlflow>=2.18.0
```

Let's spin up the MLflow tracking server, where we will store our DSPy program. The command below will start a local server at
`http://127.0.0.1:5000/`.

```bash
> mlflow ui
```

Then we can define the DSPy program and log it to the MLflow server. "log" is an overloaded term in MLflow, basically it means
we store the program information along with environment requirements in the MLflow server. See the code below:

```python
import dspy
import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("deploy_dspy_program")

lm = dspy.LM("openai/gpt-4o-mini")
dspy.settings.configure(lm=lm)
dspy_program = dspy.ChainOfThought("question -> answer")

with mlflow.start_run():
    mlflow.dspy.log_model(
        dspy_program,
        "dspy_program",
        input_example={"messages": [{"role": "user", "content": "What is LLM agent?"}]},
        task="llm/v1/chat",
    )
```

We recommend you to set `task="llm/v1/chat"` so that the deployed program automatically takes input and generate output in
the same format as the OpenAI chat API, which is a common interface for LM applications. Write the code above into
a file, e.g. `mlflow_dspy.py`, and run it.

After you logged the program, you can view the saved information in MLflow UI. Open `http://127.0.0.1:5000/` and select
the `deploy_dspy_program` experiment, then select the run your just created, under the `Artifacts` tab, you should see the
logged program information, similar to the following screenshot:

![MLflow UI](./dspy_mlflow_ui.png)

Grab your run id from UI (or the console print when you execute `mlflow_dspy.py`), now you can deploy the logged program
with the following command:

```bash
> mlflow models serve -m runs:/{run_id}/model -p 6000
```

After the program is deployed, you can test it with the following command:

```bash
> curl http://127.0.0.1:6000/invocations -H "Content-Type:application/json"  --data '{"messages": [{"content": "what is 2 + 2?", "role": "user"}]}'
```

You should see the response like below:

```json
{"choices": [{"index": 0, "message": {"role": "assistant", "content": "{\"reasoning\": \"The question asks for the sum of 2 and 2. To find the answer, we simply add the two numbers together: 2 + 2 = 4.\", \"answer\": \"4\"}"}, "finish_reason": "stop"}]}
```

For complete guide on how to deploy a DSPy program with MLflow, and how to customize the deployment, please refer to the
[MLflow documentation](https://mlflow.org/docs/latest/llms/dspy/index.html).

### Best Practices for MLflow Deployment

1. **Environment Management**: Always specify your Python dependencies in a `conda.yaml` or `requirements.txt` file.
2. **Versioning**: Use meaningful tags and descriptions for your model versions.
3. **Input Validation**: Define clear input schemas and examples.
4. **Monitoring**: Set up proper logging and monitoring for production deployments.

For production deployments, consider using MLflow with containerization:

```bash
> mlflow models build-docker -m "runs:/{run_id}/model" -n "dspy-program"
> docker run -p 6000:8080 dspy-program
```

For a complete guide on production deployment options and best practices, refer to the
[MLflow documentation](https://mlflow.org/docs/latest/llms/dspy/index.html).


## End of tutorials/deployment/index.md

---


## File: tutorials/saving/index.md

# Tutorial: Saving and Loading your DSPy program

This guide demonstrates how to save and load your DSPy program. At a high level, there are two ways to save your DSPy program:

1. Save the state of the program only, similar to weights-only saving in PyTorch.
2. Save the whole program, including both the architecture and the state, which is supported by `dspy>=2.6.0`.

## State-only Saving

State represents the DSPy program's internal state, including the signature, demos (few-shot examples), and other informaiton like
the `lm` to use for each `dspy.Predict` in the program. It also includes configurable attributes of other DSPy modules like
`k` for `dspy.retrievers.Retriever`. To save the state of a program, use the `save` method and set `save_program=False`. You can
choose to save the state to a JSON file or a pickle file. We recommend saving the state to a JSON file because it is safer and readable.
But sometimes your program contains non-serializable objects like `dspy.Image` or `datetime.datetime`, in which case you should save
the state to a pickle file.

Let's say we have compiled a program with some data, and we want to save the program for future usage:

```python
import dspy
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric

dspy.settings.configure(lm=dspy.LM("openai/gpt-4o-mini"))

gsm8k = GSM8K()
gsm8k_trainset = gsm8k.train[:10]
dspy_program = dspy.ChainOfThought("question -> answer")

optimizer = dspy.BootstrapFewShot(metric=gsm8k_metric, max_bootstrapped_demos=4, max_labeled_demos=4, max_rounds=5)
compiled_dspy_program = optimizer.compile(dspy_program, trainset=gsm8k_trainset)
```

To save the state of your program to json file:

```python
compiled_dspy_program.save("./dspy_program/program.json", save_program=False)
```

To save the state of your program to a pickle file:

```python
compiled_dspy_program.save("./dspy_program/program.pkl", save_program=False)
```

To load your saved state, you need to **recreate the same program**, then load the state using the `load` method.

```python
loaded_dspy_program = dspy.ChainOfThought("question -> answer") # Recreate the same program.
loaded_dspy_program.load("./dspy_program/program.json")

assert len(compiled_dspy_program.demos) == len(loaded_dspy_program.demos)
for original_demo, loaded_demo in zip(compiled_dspy_program.demos, loaded_dspy_program.demos):
    # Loaded demo is a dict, while the original demo is a dspy.Example.
    assert original_demo.toDict() == loaded_demo
assert str(compiled_dspy_program.signature) == str(loaded_dspy_program.signature)
```

Or load the state from a pickle file:

```python
loaded_dspy_program = dspy.ChainOfThought("question -> answer") # Recreate the same program.
loaded_dspy_program.load("./dspy_program/program.pkl")

assert len(compiled_dspy_program.demos) == len(loaded_dspy_program.demos)
for original_demo, loaded_demo in zip(compiled_dspy_program.demos, loaded_dspy_program.demos):
    # Loaded demo is a dict, while the original demo is a dspy.Example.
    assert original_demo.toDict() == loaded_demo
assert str(compiled_dspy_program.signature) == str(loaded_dspy_program.signature)
```

## Whole Program Saving

Starting from `dspy>=2.6.0`, DSPy supports saving the whole program, including the architecture and the state. This feature
is powered by `cloudpickle`, which is a library for serializing and deserializing Python objects.

To save the whole program, use the `save` method and set `save_program=True`, and specify a directory path to save the program
instead of a file name. We require a directory path because we also save some metadata, e.g., the dependency versions along
with the program itself.

```python
compiled_dspy_program.save("./dspy_program/", save_program=True)
```

To load the saved program, directly use `dspy.load` method:

```python
loaded_dspy_program = dspy.load("./dspy_program/")

assert len(compiled_dspy_program.demos) == len(loaded_dspy_program.demos)
for original_demo, loaded_demo in zip(compiled_dspy_program.demos, loaded_dspy_program.demos):
    # Loaded demo is a dict, while the original demo is a dspy.Example.
    assert original_demo.toDict() == loaded_demo
assert str(compiled_dspy_program.signature) == str(loaded_dspy_program.signature)
```

With whole program saving, you don't need to recreate the program, but can directly load the architecture along with the state.
You can pick the suitable saviing approach based on your needs.

## Backward Compatibility

As of `dspy<2.7`, we don't guarantee the backward compatibility of the saved program. For example, if you save the program with `dspy==2.5.35`,
at loading time please make sure to use the same version of DSPy to load the program, otherwise the program may not work as expected. Chances
are that loading a saved file in a different version of DSPy will not raise an error, but the performance could be different from when
the program was saved.

Starting from `dspy>=2.7`, we will guarantee the backward compatibility of the saved program in major releases, i.e., programs saved in `dspy==2.7.0`
should be loadeable in `dspy==2.7.10`.


## End of tutorials/saving/index.md

---


## File: community/community-resources.md

# Resources

This is the list of tutorials and blog posts on DSPy. If you would like to add your own tutorial, please make a PR.


## A Few Blogs & Videos on using DSPy



### Blogs

| **Name** | **Link** |
|---|---|
| **Why I bet on DSPy** | [Blog](https://blog.isaacbmiller.com/posts/dspy) |
| **Not Your Average Prompt Engineering** | [Blog](https://jina.ai/news/dspy-not-your-average-prompt-engineering/) |
| **Why I'm excited about DSPy** | [Blog](https://substack.stephen.so/p/why-im-excited-about-dspy) |
| **Achieving GPT-4 Performance at Lower Cost** | [Link](https://gradient.ai/blog/achieving-gpt-4-level-performance-at-lower-cost-using-dspy) |
| **Prompt engineering is a task best left to AI models** | [Link](https://www.theregister.com/2024/02/22/prompt_engineering_ai_models/) |
| **What makes DSPy a valuable framework for developing complex language model pipelines?** | [Link](https://medium.com/@sujathamudadla1213/what-makes-dspy-a-valuable-framework-for-developing-complex-language-model-pipelines-edfa5b4bcf9b) |
| **DSPy: A new framework to program your foundation models just by prompting** | [Link](https://www.linkedin.com/pulse/dspy-new-framework-program-your-foundation-models-just-prompting-lli4c/) |
| **Intro to DSPy: Goodbye Prompting, Hello Programming** | [Link](https://medium.com/towards-data-science/intro-to-dspy-goodbye-prompting-hello-programming-4ca1c6ce3eb9) |
| **DSPyGen: Revolutionizing AI** | [Link](https://www.linkedin.com/pulse/launch-alert-dspygen-20242252-revolutionizing-ai-sean-chatman--g9f1c/) |
| **Building an AI Assistant with DSPy** | [Link](https://www.linkedin.com/pulse/building-ai-assistant-dspy-valliappa-lakshmanan-vgnsc/) |
| **Building Self-improving Agents in Production with DSPy** | [Link](https://relevanceai.com/blog/building-self-improving-agentic-systems-in-production-with-dspy) |


### Videos
| **Name** | **Link** |
|---|---|
| **DSPy Explained! (60K views)** | [Link](https://www.youtube.com/watch?v=41EfOY0Ldkc) |
| **DSPy Intro from Sephora (25K views)** | [Link](https://www.youtube.com/watch?v=D2HurSldDkE) |
| **Structured Outputs with DSPy** | [Link](https://www.youtube.com/watch?v=tVw3CwrN5-8) |
| **DSPy and ColBERT - Weaviate Podcast** | [Link](https://www.youtube.com/watch?v=CDung1LnLbY) |
| **SBTB23 DSPy** | [Link](https://www.youtube.com/watch?v=Dt3H2ninoeY) |
| **Optimization with DSPy and LangChain** | [Link](https://www.youtube.com/watch?v=4EXOmWeqXRc) |
| **Automated Prompt Engineering + Visualization** | [Link](https://www.youtube.com/watch?v=eAZ2LtJ6D5k) |
| **Transforming LM Calls into Pipelines** | [Link](https://www.youtube.com/watch?v=NoaDWKHdkHg) |
| **NeurIPS Hacker Cup: DSPy for Code Gen** | [Link](https://www.youtube.com/watch?v=gpe-rtJN8z8) |
| **MIPRO and DSPy - Weaviate Podcast** | [Link](https://www.youtube.com/watch?v=skMH3DOV_UQ) |
| **Getting Started with RAG in DSPy** | [Link](https://www.youtube.com/watch?v=CEuUG4Umfxs) |
| **Adding Depth to DSPy Programs** | [Link](https://www.youtube.com/watch?v=0c7Ksd6BG88) |
| **Programming Foundation Models with DSPy** | [Link](https://www.youtube.com/watch?v=Y94tw4eDHW0) |
| **DSPy End-to-End: SF Meetup** | [Link](https://www.youtube.com/watch?v=Y81DoFmt-2U) |
| **Monitoring & Tracing DSPy with Langtrace** | [Link](https://langtrace.ai/blog/announcing-dspy-support-in-langtrace) |
| **Teaching chat models to solve chess puzzles using DSPy + Finetuning** | [Link](https://raw.sh/posts/chess_puzzles) |
| **Build Self-Improving AI Agents with DSPy (No Code)** | [Link](https://www.youtube.com/watch?v=UY8OsMlV21Y) |

### Podcasts

Weaviate has a directory of 10 amazing notebooks and 6 podcasts!
Huge shoutout to them for the massive support ❤️. See the [Weaviate DSPy directory](https://weaviate.io/developers/weaviate/more-resources/dspy).


TODO: This list in particular is highly incomplete. There are dozens of other good ones. To allow space, divide into opintionated blogs / podcasts / interviews vs. tutorials & talks.

Credit: Some of these resources were originally compiled in the [Awesome DSPy](https://github.com/ganarajpr/awesome-dspy/tree/master) repo.



## End of community/community-resources.md

---


## File: community/how-to-contribute.md

## Development Setup

We are working on a development setup guide. If you're interested in contributing, please reach out to us on Discord.

## Contributing

**What if I have a better idea for prompting or synthetic data generation?** Perfect. We encourage you to think if it's best expressed as a module or an optimizer, and we'd love to merge it in DSPy so everyone can use it. DSPy is not a complete project; it's an ongoing effort to create structure (modules and optimizers) in place of hacky prompt and pipeline engineering tricks.

**How can I add my favorite LM or vector store?** 

Check out these walkthroughs on setting up a [Custom LM client](/deep-dive/language_model_clients/custom-lm-client) and [Custom RM client](/deep-dive/retrieval_models_clients/custom-rm-client).



## End of community/how-to-contribute.md

---


## File: community/use-cases.md

# Use Cases

We often get questions like "How are people using DSPy in practice?", both in production and for research. This list was created to collect a few pointers and to encourage others in the community to add their own work below.

This list is ever expanding and highly incomplete (WIP)! We'll be adding a bunch more. If you would like to add your product or research to this list, please make a PR.

## A Few Company Use Cases

| **Name** | **Use Cases** |
|---|---|
| **[JetBlue](https://www.jetblue.com/)** | Multiple chatbot use cases. [Blog](https://www.databricks.com/blog/optimizing-databricks-llm-pipelines-dspy) |
| **[Replit](https://replit.com/)** | Synthesize diffs using code LLMs using a DSPy pipeline. [Blog](https://blog.replit.com/code-repair) |
| **[Databricks](https://www.databricks.com/)** | Research, products, and customer solutions around LM Judges, RAG, classification, and other applications. [Blog](https://www.databricks.com/blog/dspy-databricks) [Blog II](https://www.databricks.com/customers/ddi) |
| **[Sephora](https://www.sephora.com/)** | Undisclosed agent usecases; perspectives shared in [DAIS Session](https://www.youtube.com/watch?v=D2HurSldDkE). |
| **[Zoro UK](https://www.zoro.co.uk/)** | E-commerce applications around structured shopping. [Portkey Session](https://www.youtube.com/watch?v=_vGKSc1tekE) |
| **[VMware](https://www.vmware.com/)** | RAG and other prompt optimization applications. [Interview in The Register.](https://www.theregister.com/2024/02/22/prompt_engineering_ai_models/) [Business Insider.](https://www.businessinsider.com/chaptgpt-large-language-model-ai-prompt-engineering-automated-optimizer-2024-3) |
| **[Haize Labs](https://www.haizelabs.com/)** | Automated red-teaming for LLMs. [Blog](https://blog.haizelabs.com/posts/dspy/) |
| **[Plastic Labs](https://www.plasticlabs.ai/)** | Different pipelines within Honcho. [Blog](https://blog.plasticlabs.ai/blog/User-State-is-State-of-the-Art) |
| **[PingCAP](https://pingcap.com/)** | Building a knowledge graph. [Article](https://www.pingcap.com/article/building-a-graphrag-from-wikipedia-page-using-dspy-openai-and-tidb-vector-database/) |
| **[Salomatic](https://langtrace.ai/blog/case-study-how-salomatic-used-langtrace-to-build-a-reliable-medical-report-generation-system)** | Enriching medical reports using DSPy. [Blog](https://langtrace.ai/blog/case-study-how-salomatic-used-langtrace-to-build-a-reliable-medical-report-generation-system) |
| **[Truelaw](https://www.youtube.com/watch?v=O0F3RAWZNfM)** | How Truelaw builds bespoke LLM pipelines for law firms using DSPy. [Podcast](https://www.youtube.com/watch?v=O0F3RAWZNfM) |
| **[Moody's](https://www.moodys.com/)** | Leveraging DSPy to optimize RAG systems, LLM-as-a-Judge, and agentic systems for financial workflows. |
| **[Normal Computing](https://www.normalcomputing.com/)** | Translate specs from chip companies from English to intermediate formal languages |
| **[Procure.FYI](https://www.procure.fyi/)** | Process messy, publicly available technology spending and pricing data via DSPy. |
| **[RadiantLogic](https://www.radiantlogic.com/)** | AI Data Assistant. DSPy is used for the agent that routes the query, the context extraction module, the text-to-sql conversion engine, and the table summarization module. |
| **[Raia](https://raiahealth.com/)** | Using DSPy for AI-powered Personal Healthcare Agents. |
| **[Hyperlint](https://hyperlint.com)** | Uses DSPy to generate technical documentation. DSPy helps to fetch relevant information and synthesize that into tutorials. |
| **[Starops](https://staropshq.com/) & [Saya](https://heysaya.ai/)** | Building research documents given a user's corpus. Generate prompts to create more articles from example articles. |
| **[Tessel AI](https://tesselai.com/)** | Enhancing human-machine interaction with data use cases. |
| **[Dicer.ai](https://dicer.ai/)** | Uses DSPy for marketing AI to get the most from their paid ads. |
| **[Howie](https://howie.ai)** | Using DSPy to automate meeting scheduling through email. |
| **[Isoform.ai](https://isoform.ai)** | Building custom integrations using DSPy. |
| **[Trampoline AI](https://trampoline.ai)** | Uses DSPy to power their data-augmentation and LM pipelines. |
| **[Pretrain](https://pretrain.com)** | Uses DSPy to automatically optimize AI performance towards user-defined tasks based on uploaded examples. |

WIP. This list mainly includes companies that have public posts or have OKed being included for specific products so far.


## A Few Papers Using DSPy

| **Name** | **Description** |
|---|---|
| **[STORM](https://arxiv.org/abs/2402.14207)** | Writing Wikipedia-like Articles From Scratch. |
| **[PATH](https://arxiv.org/abs/2406.11706)** | Prompts as Auto-Optimized Training Hyperparameters: Training Best-in-Class IR Models from Scratch with 10 Gold Labels |
| **[WangLab @ MEDIQA](https://arxiv.org/abs/2404.14544)** | UofT's winning system at MEDIQA, outperforms the next best system by 20 points |
| **[UMD's Suicide Detection System](https://arxiv.org/abs/2406.06608)** | Outperforms 20-hour expert human prompt engineering by 40% |
| **[IReRa](https://arxiv.org/abs/2401.12178)** | Infer-Retrieve-Rank: Extreme Classification with > 10,000 Labels |
| **[Unreasonably Effective Eccentric Prompts](https://arxiv.org/abs/2402.10949v2)** | General Prompt Optimization |
| **[Palimpzest](https://arxiv.org/abs/2405.14696)** | A Declarative System for Optimizing AI Workloads |
| **[AI Agents that Matter](https://arxiv.org/abs/2407.01502v1)** | Agent Efficiency Optimization |
| **[EDEN](https://arxiv.org/abs/2406.17982v1)** | Empathetic Dialogues for English Learning: Uses adaptive empathetic feedback to improve student grit |
| **[ECG-Chat](https://arxiv.org/pdf/2408.08849)** | Uses DSPy with GraphRAG for medical report generation |
| **[DSPy Assertions](https://arxiv.org/abs/2312.13382)** | Various applications of imposing hard and soft constraints on LM outputs |
| **[DSPy Guardrails](https://boxiyu.github.io/assets/pdf/DSPy_Guardrails.pdf)** | Reduce the attack success rate of CodeAttack, decreasing from 75% to 5% |
| **[Co-STORM](https://arxiv.org/pdf/2408.15232)** | Collaborative STORM: Generate Wikipedia-like articles through collaborative discourse among users and multiple LM agents |

## A Few Repositories (or other OSS examples) using DSPy

| **Name** | **Description/Link** |
|---|---|
| **Stanford CS 224U Homework** | [Github](https://github.com/cgpotts/cs224u/blob/main/hw_openqa.ipynb) |
| **STORM Report Generation (10,000 GitHub stars)** | [Github](https://github.com/stanford-oval/storm) |
| **DSPy Redteaming** | [Github](https://github.com/haizelabs/dspy-redteam) |
| **DSPy Theory of Mind** |  [Github](https://github.com/plastic-labs/dspy-opentom) |
| **Indic cross-lingual Natural Language Inference** |  [Github](https://github.com/saifulhaq95/DSPy-Indic/blob/main/indicxlni.ipynb) |
| **Optimizing LM for Text2SQL using DSPy** | [Github](https://github.com/jjovalle99/DSPy-Text2SQL) |
| **DSPy PII Masking Demo by Eric Ness** | [Colab](https://colab.research.google.com/drive/1KZR1sGTp_RLWUJPAiK1FKPKI-Qn9neUm?usp=sharing) |
| **DSPy on BIG-Bench Hard Example** |  [Github](https://drchrislevy.github.io/posts/dspy/dspy.html) |
| **Building a chess playing agent using DSPy** |  [Github](https://medium.com/thoughts-on-machine-learning/building-a-chess-playing-agent-using-dspy-9b87c868f71e) |
| **Ittia Research Fact Checking** | [Github](https://github.com/ittia-research/check) |
| **Strategic Debate via Tree-of-Thought** | [Github](https://github.com/zbambergerNLP/strategic-debate-tot) |
| **Sanskrit to English Translation App**| [Github](https://github.com/ganarajpr/sanskrit-translator-dspy) |
| **DSPy for extracting features from PDFs on arXiv**| [Github](https://github.com/S1M0N38/dspy-arxiv) |
| **DSPygen: DSPy in Ruby on Rails**| [Github](https://github.com/seanchatmangpt/dspygen) |
| **DSPy Inspector**| [Github](https://github.com/Neoxelox/dspy-inspector) |
| **DSPy with FastAPI**| [Github](https://github.com/diicellman/dspy-rag-fastapi) |
| **DSPy for Indian Languages**| [Github](https://github.com/saifulhaq95/DSPy-Indic) |
| **Hurricane: Blog Posts with Generative Feedback Loops!**| [Github](https://github.com/weaviate-tutorials/Hurricane) |
| **RAG example using DSPy, Gradio, FastAPI, and Ollama**| [Github](https://github.com/diicellman/dspy-gradio-rag) |
| **Synthetic Data Generation**| [Github](https://colab.research.google.com/drive/1CweVOu0qhTC0yOfW5QkLDRIKuAuWJKEr?usp=sharing) |
| **Self Discover**| [Github](https://colab.research.google.com/drive/1GkAQKmw1XQgg5UNzzy8OncRe79V6pADB?usp=sharing) |

TODO: This list in particular is highly incomplete. There are a couple dozen other good ones.

## A Few Providers, Integrations, and related Blog Releases

| **Name** | **Link** |
|---|---|
| **Databricks** | [Link](https://www.databricks.com/blog/dspy-databricks) |
| **Zenbase** | [Link](https://zenbase.ai/) |
| **LangWatch** | [Link](https://langwatch.ai/blog/introducing-dspy-visualizer) |
| **Gradient** | [Link](https://gradient.ai/blog/achieving-gpt-4-level-performance-at-lower-cost-using-dspy) |
| **Snowflake** | [Link](https://medium.com/snowflake/dspy-snowflake-140d6d947d73) |
| **Langchain** | [Link](https://python.langchain.com/v0.2/docs/integrations/providers/dspy/) |
| **Weaviate** | [Link](https://weaviate.io/blog/dspy-optimizers) |
| **Qdrant** | [Link](https://qdrant.tech/documentation/frameworks/dspy/) |
| **Weights & Biases Weave** | [Link](https://weave-docs.wandb.ai/guides/integrations/dspy/) |
| **Milvus** | [Link](https://milvus.io/docs/integrate_with_dspy.md) |
| **Neo4j** | [Link](https://neo4j.com/labs/genai-ecosystem/dspy/) |
| **Lightning AI** | [Link](https://lightning.ai/lightning-ai/studios/dspy-programming-with-foundation-models) |
| **Haystack** | [Link](https://towardsdatascience.com/automating-prompt-engineering-with-dspy-and-haystack-926a637a3f43) |
| **Arize** | [Link](https://docs.arize.com/phoenix/tracing/integrations-tracing/dspy) |
| **LlamaIndex** | [Link](https://github.com/stanfordnlp/dspy/blob/main/examples/llamaindex/dspy_llamaindex_rag.ipynb) |
| **Langtrace** | [Link](https://docs.langtrace.ai/supported-integrations/llm-frameworks/dspy) |
| **Langfuse** | [Link](https://langfuse.com/docs/integrations/dspy) |
| **OpenLIT** | [Link](https://docs.openlit.io/latest/integrations/dspy) |
| **Relevance AI** | [Link](https://relevanceai.com/blog/dspy-programming---not-prompting---language-models) |

Credit: Some of these resources were originally compiled in the [Awesome DSPy](https://github.com/ganarajpr/awesome-dspy/tree/master) repo.


## End of community/use-cases.md

---
