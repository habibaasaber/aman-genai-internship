# Prompt Stress-Test Suite Results

## Executive Summary
This report compares two providers across three prompt strategies and three business tasks using a simple heuristic accuracy score plus cost and latency metrics.

### Overall Metrics
| Model            |   Latency_sec |   Estimated_Cost_USD |   Total_Tokens |   Accuracy_Score |   Overall_Score |
|:-----------------|--------------:|---------------------:|---------------:|-----------------:|----------------:|
| gemini_2_0_flash |       2.65851 |             0.001358 |          12045 |         0.144444 |        0.286667 |
| gpt_4o_mini      |       1.63162 |             0.002886 |          12449 |         0.733333 |        0.64     |

## Task Winners
| Task   | Winner      |   Overall_Score |   Accuracy |   Cost_Score |   Latency_Score | Reason                                                                                           |
|:-------|:------------|----------------:|-----------:|-------------:|----------------:|:-------------------------------------------------------------------------------------------------|
| task1  | gpt_4o_mini |            0.6  |      0.667 |            0 |               1 | It produced the most reliable JSON structure while keeping latency and cost competitive.         |
| task2  | gpt_4o_mini |            0.7  |      0.833 |            0 |               1 | It extracted PII entities most consistently and stayed efficient under the evaluation heuristic. |
| task3  | gpt_4o_mini |            0.62 |      0.7   |            0 |               1 | It drafted the most complete, empathetic, and actionable reply while remaining cost-effective.   |

## Visualizations
### Cost Comparison
![Cost Comparison](outputs/charts/cost_comparison.png)

### Latency Comparison
![Latency Comparison](outputs/charts/latency_comparison.png)

## JSON Output Reliability (Task 1)
Percentage of valid JSON outputs:

| Model            | Strategy         |   Success Rate (%) |
|:-----------------|:-----------------|-------------------:|
| gemini_2_0_flash | chain_of_thought |                  0 |
| gemini_2_0_flash | few_shot         |                  0 |
| gemini_2_0_flash | zero_shot        |                  0 |
| gpt_4o_mini      | chain_of_thought |                100 |
| gpt_4o_mini      | few_shot         |                100 |
| gpt_4o_mini      | zero_shot        |                100 |
