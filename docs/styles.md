# Style Guide

## 1. Hands-On Machine Learning

### 1.1 Types of Machine Learning Systems

- Use supervised learning for classification or regression when training examples include labels or target values.
- Use unsupervised learning to discover structure such as clusters, lower-dimensional representations, or anomalies.
- Use semi-supervised learning when a small labeled set can guide learning from a much larger unlabeled set.
- Use self-supervised learning to derive targets from unlabeled data before transferring the representation to a task.
- Use reinforcement learning when an agent must learn a policy that maximizes cumulative reward through interaction.
- Use batch learning when periodically retraining from the full dataset can keep the model sufficiently current.
- Use online learning when the model must adapt incrementally or learn from data that cannot fit in memory at once.
- Use instance-based learning when predictions should depend on a meaningful similarity to stored examples.
- Use model-based learning when a fitted model should generalize patterns from training data to unseen instances.

### 1.2 Main Challenges of Machine Learning

- Obtain enough training data for the task's complexity, and compare the value of more data, algorithm improvements,
  and transfer learning before investing in any one approach.
- Make the training sample representative of production cases, accounting for sampling noise, sampling bias, and
  nonresponse bias in how the data was collected.
- Investigate errors, outliers, noise, and missing values, then document whether each issue is corrected, removed, or
  imputed.
- Improve the signal in the inputs through feature selection, feature extraction, or collection of new features rather
  than retaining irrelevant attributes by default.
- Treat low training error with high validation error as overfitting, and respond by simplifying or constraining the
  model, reducing irrelevant features, adding data, or cleaning noise.
- Treat poor performance even on the training data as underfitting, and respond with a more expressive model, better
  features, or fewer constraints.
- Consider maintainability, model size, latency, scalability, security, model aging, and retraining needs before
  choosing a model for deployment.

### 1.3 Testing and Validating

- Reserve a test set to estimate generalization error on unseen cases, and size it by the number of examples needed
  for a reliable estimate rather than by a fixed percentage alone.
- Compare training and validation errors to identify overfitting before using the test set.
- Select models and tune hyperparameters on a validation set, not on the test set.
- After model selection, retrain the chosen model with the available training and validation data, then evaluate it once
  on the test set.
- Balance validation-set size against evaluation uncertainty and loss of training data; use repeated cross-validation
  when a more precise estimate justifies the additional training cost.
- Make validation and test sets representative of production data, and prevent duplicates or near-duplicates from being
  split across them.
- When training and production distributions differ, use a train-development set from the training distribution to
  distinguish overfitting from data mismatch before changing the data or model.
- State the assumptions behind each candidate model and compare a reasonable set of candidates under the same
  conditions instead of assuming that one model family is always best.

### 1.4 Look at the Big Picture

- Frame the problem by defining the business objective, current baseline, downstream use, pipeline context, prediction
  task, and learning requirements before selecting a model.
- Select a performance measure that reflects the objective and the error distribution, including the importance of large
  errors and sensitivity to outliers.
- Check assumptions about the data, targets, and downstream use with relevant stakeholders before implementation.

### 1.5 Get the Data

- Automate authorized data acquisition, extraction, and loading so the same procedure can process refreshed data.
- Inspect row counts, feature types, missing values, category frequencies, summary statistics, distributions, prior
  transformations, and capped values before modeling.
- Create and isolate a stable test set before detailed exploration, using immutable identifiers when possible and
  stratified sampling when important population proportions must be preserved.

### 1.6 Explore and Visualize the Data to Gain Insights

- Visualize only the training data, using domain-appropriate encodings for location, density, magnitude, and targets to
  reveal meaningful patterns.
- Look for correlations with numerical summaries and plots, while checking for nonlinear relationships, caps, artifacts,
  and outliers that a linear coefficient can miss.
- Experiment with domain-informed attribute combinations, verify that they add useful information without excessive
  collinearity, and revisit them after analyzing model errors.

### 1.7 Prepare the Data for Machine Learning Algorithms

- Clean a fresh copy of the training data by separating inputs from targets and choosing whether to remove or impute
  missing values, learning any imputation statistics from the training set only.
- Handle text and categorical attributes by distinguishing ordinal from nominal values, preserving the learned output
  schema, defining unseen-category behavior, and controlling high-cardinality costs.
- Scale and transform features according to model requirements and distribution shape, fitting parameters on training
  data only and including an inverse transformation when the target is transformed.
- Implement custom transformations with the appropriate stateless or trainable interface, explicit validation, stable
  feature metadata, reproducible randomness, and an inverse operation when one exists.
- Compose transformations in a fixed pipeline by column type, preserving row alignment and output schema, managing
  dense and sparse representations, and fitting preprocessing with the model to prevent leakage.

### 1.8 Select and Train a Model

- Train and evaluate a simple end-to-end baseline on the training set, using the chosen metric to identify underfitting
  and treating implausibly low training error as a possible sign of overfitting.
- Use cross-validation within the training data to compare diverse candidates under the same conditions, report score
  level and variation, account for computation cost, and shortlist a few promising models.

### 1.9 Fine-Tune Your Model

- Use grid search for a manageable set of discrete combinations, tuning interacting preprocessing and model settings
  together and expanding ranges when the best value lies on a boundary.
- Use randomized search with suitable value distributions and a fixed computation budget for large or continuous search
  spaces, optionally allocating increasing resources to promising candidates.
- Build an ensemble from strong models with different error patterns, and keep it only when it improves validation
  performance over its members.
- Analyze the best models, feature importance, individual errors, and subgroup fairness to guide data cleaning, feature
  revision, usage restrictions, or further modeling.
- Evaluate the finalized pipeline once on the isolated test set, quantify uncertainty when needed, avoid tuning to the
  result, and communicate reproducible findings, assumptions, and limitations to the intended audience.

### 1.10 Launch, Monitor, and Maintain Your System

- Deploy the complete preprocessing and model pipeline through a stable interface, monitor prediction and input quality,
  automate safe retraining and comparison, and retain versioned models and data for rapid rollback.

## 2. Software Engineering for Data Scientists

### 2.1 What Is Good Code?

- Keep code simple by avoiding repetition, accidental complexity, and unnecessary work.
- Divide code into logical, self-contained components with clear inputs, outputs, and responsibilities.
- Optimize for readers through consistent formatting, descriptive names, focused documentation, and straightforward
  control flow.
- Make code only as efficient as its requirements demand, without exceeding available time or memory resources.
- Make results reproducible, report useful errors, and handle expected invalid or unusual inputs deliberately.
- Revisit rushed prototypes when they become reusable or long-lived so that temporary compromises do not become
  permanent complexity.

### 2.2 Analyzing Code Performance

- Optimize only when measured performance does not meet a concrete requirement.
- Measure before changing code, then use timing or profiling information to locate the actual bottleneck.
- Consider how runtime and memory use will scale as the data grows, not only how the code performs on a small sample.
- Compare alternatives with representative inputs and preserve readability unless the measured benefit justifies added
  complexity.

### 2.3 Using Data Structures Effectively

- Choose a data structure according to the operations the code performs most often, such as indexed access, keyed
  lookup, membership testing, multidimensional calculation, or tabular analysis.
- Prefer built-in data structures when they solve the problem clearly and efficiently.
- Use vectorized operations for homogeneous numerical arrays and tabular data when they are available, rather than
  repeatedly processing individual elements or rows.
- Select appropriate data types and sparse or out-of-memory representations when data size makes memory use material.
- Measure candidate structures with representative data instead of assuming that a theoretically suitable structure is
  fastest in the actual workflow.

### 2.4 Object-Oriented Programming and Functional Programming

- Use classes when multiple similar objects need to keep related state and behavior together.
- Do not introduce a class for a single instance or when a small group of functions expresses the problem more simply.
- Keep class interfaces consistent, limit shared mutable state, and use inheritance only when the relationship is clear.
- Prefer functions that make inputs, outputs, and data transformations explicit when state does not need to change.
- Choose a programming paradigm because it fits the problem and the surrounding code, not to satisfy a pattern in
  isolation.

### 2.5 Errors, Logging, and Debugging

- Read error messages from the final reported failure back through the call path to identify the relevant cause.
- Catch an exception only when the code can recover, add meaningful context, or perform required cleanup.
- Raise specific errors with messages that explain what was invalid and what the caller can do about it.
- Record diagnostics when execution history, severity, or operational visibility must be preserved; do not use ad hoc
  output as a permanent substitute.
- Keep logging configuration at the application entry point and keep reusable library code independent of output
  destinations and presentation choices.
- Debug systematically by reproducing the problem, narrowing its location, inspecting relevant state, and verifying the
  correction.

### 2.6 Code Formatting, Linting, and Type Checking

- Follow PEP 8 and the established project conventions unless an intentional local rule overrides them.
- Use automated formatting and import organization to keep style consistent without manual cleanup.
- Use static analysis to identify likely defects, confusing constructs, and inconsistent conventions before execution.
- Add type annotations where they clarify interfaces and allow incorrect assumptions to be detected earlier.
- Run formatting, linting, and type checks consistently during development and automate them where practical.

### 2.7 Testing Your Code

- Test stable behavior so later changes can be made with confidence and previous defects do not return.
- Structure each test around setup, one relevant action, explicit assertions, and cleanup when cleanup is required.
- Cover representative inputs, boundaries, invalid inputs, and important failure behavior.
- Separate focused component tests from tests that verify interactions between components.
- Validate important data properties and schemas at boundaries where unexpected data would invalidate later results.
- For machine learning workflows, test data preparation, training and inference interfaces, expected output properties,
  and a small set of predictable examples even when exact trained parameters vary.
- Run tests through an automated test runner rather than relying on manual checks.

### 2.8 Design and Refactoring

- Design around the project's purpose, expected users, inputs, outputs, constraints, and likely execution environment.
- Give every component one clear responsibility and define its interface before filling in implementation details.
- Keep coupling low so one component can change without requiring unrelated changes elsewhere.
- Use notebooks for exploration, prototyping, and explanatory work; move reusable or repeatedly executed behavior into
  importable modules.
- Before extracting notebook code, confirm that the notebook runs from start to finish and identify cells that form a
  coherent reusable operation.
- Treat repeatedly restarting a notebook from the middle as a signal that the workflow may need separate stages or
  reusable functions.
- Refactor in small steps, preserve externally observable behavior, and run the relevant tests after each step.

### 2.9 Documentation

- Write documentation for its intended audience and keep it current with the code and analytical decisions.
- Record why data, methods, assumptions, and tradeoffs were chosen, including limitations and unsuccessful approaches
  that would help future work.
- Let names communicate purpose, use comments for context or caveats not evident from the code, and use docstrings to
  describe public behavior, inputs, outputs, and relevant edge cases.
- Keep each fact in one appropriate place rather than duplicating it across comments, docstrings, and longer documents.
- Give every notebook a descriptive filename, explain its purpose at the start, organize it with meaningful headings,
  and interleave Markdown with the code it explains.
- Use notebook Markdown for summaries, assumptions, caveats, decisions, and interpretations rather than restating
  visible code operations.
- Track experiment data, splits, feature choices, parameters, evaluation measures, assumptions, and results in a
  structured and reproducible form.

### 2.10 Sharing Your Code: Version Control, Dependencies, and Packaging

- Keep source changes in version control with focused commits that explain the purpose of each change.
- Use branches and reviews to isolate work and discuss changes before integration.
- Work in an isolated environment and record the dependencies and compatible versions required to reproduce the project.
- Structure reusable code as an installable package when other projects need to import it.
- Keep package metadata, build configuration, documentation, and license information aligned with the distributed code.

### 2.11 APIs

- Use an API when functionality or data must be shared across processes or systems through a stable interface.
- Define endpoints around clear resources or operations, and document request data, response data, status behavior, and
  errors.
- Validate input at the boundary and return errors that help callers correct their requests without exposing sensitive
  implementation details.
- Keep domain logic separate from the transport layer so it can be tested and reused independently.
- Treat external API responses as untrusted input and handle unsuccessful responses, missing data, and compatibility
  changes explicitly.

### 2.12 Automation and Deployment

- Automate repeated formatting, analysis, testing, building, and deployment steps when automation reduces mistakes and
  improves reproducibility.
- Keep development, testing, and deployed environments sufficiently separate to prevent experimental work from affecting
  users.
- Require automated checks to pass before building or deploying a revision.
- Package runtime code and dependencies so the deployed environment can be recreated consistently.
- Make deployment observable and reversible, and account for operational cost and resource cleanup.

### 2.13 Security

- Consider security throughout design, development, testing, deployment, and maintenance rather than only at release.
- Keep credentials and sensitive data out of source code, notebooks, logs, generated artifacts, and version control.
- Review third-party dependencies and serialized data formats before trusting or executing their contents.
- Validate external data and model inputs, restrict access to exposed interfaces, and limit abusive or unexpected use.
- Test important adversarial or malformed inputs and monitor deployed systems for behavior outside expected ranges.
- Maintain a practical process for updating, redeploying, or rolling back code and models when a vulnerability is found.

### 2.14 Working in Software

- Organize work across planning, design, implementation, testing, deployment, and maintenance, and revisit these stages
  iteratively as requirements and findings change.
- Break research work into small, verifiable increments that can incorporate feedback and new evidence.
- Make ownership, interfaces, assumptions, and handoffs clear when collaborating across engineering, data, product,
  quality, and design roles.
- Use reviews to share context, improve code, and identify risks rather than treating them only as approval gates.
- Learn from the wider software and data communities while evaluating external practices against this project's needs.

### 2.15 Next Steps

- Improve code pragmatically: apply the rigor appropriate to its lifetime, users, risk, and likelihood of reuse.
- Standardize recurring work, place complexity behind clear interfaces, and automate repeated processes when the benefit
  is demonstrated.
- Review and test code produced with automated assistance as carefully as code written manually.
- Keep learning as practices change, seek feedback, share useful knowledge, and consider the consequences of the systems
  being built.
