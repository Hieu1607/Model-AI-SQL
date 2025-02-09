# text2sql_challenge

Challenge: Build an AI model to generate SQL query from given user requirement and context.

## Dataset and Evaluation

### Dataset

You can get the train and test set in [here](./dataset)

- 105,851 records partitioned into 100,000 train and 5,851 test records
- Database context, including table and view create statements
- Natural language explanations of what the SQL query is doing

```
|-- id: integer                            Data record id
|-- domain: string                         Domain/vertical (e.g., aerospace)
|-- domain_description: string             Domain/vertical data description
|-- sql_complexity: string                 SQL code complexity label
|-- sql_complexity_description: string     SQL code complexity description
|-- sql_task_type: string                  Type of SQL task being performed
|-- sal_task_type_description: string      Description of the SQL task type
|-- sql_prompt: string                     Natural language SQL prompt
|-- sql_context: string                    Context for the SQL prompt
|-- sql: string                            SQL corresponding to the SQL prompt
|-- sql_explanation: string                Explanation of the SQL query
```

*Note: The output evaluation will use the `sql` attribute to calculate BLEU score*

*Examples:*

```json
{
  "id": 39325,
  "domain": "public health",
  "domain_description": "Community health statistics, infectious disease tracking data, healthcare access metrics, and public health policy analysis.",
  "sql_complexity": "aggregation",
  "sql_complexity_description": "aggregation functions (COUNT, SUM, AVG, MIN, MAX, etc.), and HAVING clause",
  "sql_task_type": "analytics and reporting",
  "sql_task_type_description": "generating reports, dashboards, and analytical insights",
  "sql_prompt": "What is the total number of hospital beds in each state?",
  "sql_context": "CREATE TABLE Beds (State VARCHAR(50), Beds INT); INSERT INTO Beds (State, Beds) VALUES ('California', 100000), ('Texas', 85000), ('New York', 70000);",
  "sql": "SELECT State, SUM(Beds) FROM Beds GROUP BY State;",
  "sql_explanation": "This query calculates the total number of hospital beds in each state in the Beds table. It does this by using the SUM function on the Beds column and grouping the results by the State column."
}
```

### Evaluation

BLEU-score: Evaluate the alignment between a machine's output and that of a human.

How to use? See [here](https://huggingface.co/spaces/evaluate-metric/bleu) for more details.

```python
# pip install evaluate
>>> predictions = ["SELECT * FROM tbo.CUSTOMERS", "SELECT TenSV, MSSV FROM QLDT"]
>>> references = [
...     ["SELECT * FROM CUSTOMERS"],
...     ["SELECT * FROM QLDT"]
... ]
>>> bleu = evaluate.load("bleu")
>>> results = bleu.compute(predictions=predictions, references=references)
>>> print(results)
```

## Submission

Submit a `json` file named `results.json` following the format provided below:

```jsonlines
{'id': 1, 'answer': 'SELECT * FROM tbo.CUSTOMERS'}
{'id': 2, 'answer': 'SELECT * QLDT'}
...
```

**!!!IMPORTANT**: Use only the provided training data to build the model. **Do not use** any external data. Your submission need to at least surpass the baseline score.

*Please send the submission to me via direct message on Teams*


## Leaderboard

Your submission score will be shown in the leaderboard below. The highest score is in bold.

|StudentID|Score|Best score|
|:---|:---|:---|
|20224983|0| |
|20224899|0| |
|20235049|0| |
|20224948|0| |
|20235139|0| |
|20235260|0| |
|Baseline|to be update| |

## Deadline: 23h59 23/12/2024
