# Employee Attrition Analysis

### Summary

- The dataset was small and there were some columns that were not clearly distinguishable like `MonthlyRate` and `MonthlyIncome`. Most of the columns were mainly for exploratory purposes. Another note of the data set is that, most of the collected data was biased in the sense that there were more bachelors compared to phd employees, more males compared females, and departments where about 80% of employees work at. 
- Models used were Decision Trees, Random Forests and Neural Networks. Out of all of them **Random Forest** performed the best.


- Accuracy of Decision Tree: 86%
- Accuracy of Random Forest: 87%
- Accuracy of Neural Network : 82%
- The model that should be deployed is Random Forest as it has a higher ROC curve.
- Future work would be to use more features to test accuracies or to increase the data collection regarding to attrition. 



#### Feature Descriptions 

| Features                      | Description                                                                                                                   |
|-------------------------------|-------------------------------------------------------------------------------------------------------------------------------|
|``Age``                        | The age of employee.                                                                                                          |
|``Attrition``                  | Whether the employee left the company or not.                                                                                 |
|``BusinessTravel``             | The frequency or type of business-related travel that an employee undertakes.                                                 |
|``DailyRate``                  | The daily rate of pay for an employee                                                                                         |
|``Department``                 | The department or division within a company where the employee works.                                                         |
|``DistanceFromHome``           | Distance in miles from the employee's home and their workplace.                                                               |
|``Education``                  | The highest level of education completed by the employee.                                                                     |
|``EducationField``             | The field or area of study in which the employee's education is focused.                                                      |
|``EmployeeCount``              | The number of employees.                                                                                                      |
|``EmployeeNumber``             | A unique identifier of the employee.                                                                                          |
|``EnvironmentSatisfaction``    | The level of satisfaction or contentment an employee has with their work environment.                                         |
|``Gender``                     | Gender identity of the employee.                                                                                              |
|``HourlyRate``                 | The hourly rate of pay for an employee                                                                                        |
|``JobInvolvement``             | The degree to which an employee is engaged and involved in their job tasks and responsibilities.                              |
|``JobLevel``                   | Rank of the employee within the company.                                                                                      |
|``JobRole``                    | Position that the employee holds within their department or team.                                                             |
|``JobSatisfaction``            | The level of satisfaction or contentment an employee has with their job.                                                      |
|``MaritalStatus``              | The marital status of the employee.                                                                                           |
|``MonthlyIncome``              | The monthly income or salary earned by the employee.                                                                          |
|``MonthlyRate``                | ?????                                                                                                                         |
|``NumCompaniesWorked``         | The number of different companies that the employee has worked for.                                                           |
|``Over18``                     | Whether the employee is over 18 years old.                                                                                    |
|``OverTime``                   | Whether the employee works overtime hours.                                                                                    |
|``PercentSalaryHike``          | The percentage increase in salary that an employee received.                                                                  |
|``PerformanceRating``          | The rating or evaluation of an employee's performance.                                                                        |
|``RelationshipSatisfaction``   | The level of satisfaction or contentment an employee has with their relationships at work.                                    |
|``StandardHours``              | The standard number of hours worked per week or per day.                                                                      |
|``StockOptionLevel``           | The level or amount of stock options granted to an employee as part of their compensation package.                            |
|``TotalWorkingYears``          | The total number of years that an employee has been working.                                                                  |
|``TrainingTimesLastYear``      | The number of training sessions attended by the employee in the last year.                                                    |
|``WorkLifeBalance``            | The balancebetween an employee's work responsibilities and their personal life.                                               |
|``YearsAtCompany``             | Number of years at the company.                                                                                               |
|``YearsInCurrentRole``         | The number of years that an employee has been in their current role.                                                          |
|``YearsSinceLastPromotion``    | The number of years since the employee's last promotion.                                                                      |
|``YearsWithCurrManager``       | The number of years that an employee has been under their current manager's supervision.                                      |


### Sources
- https://towardsdatascience.com/100-stacked-charts-in-python-6ca3e1962d2b
- https://medium.com/geekculture/the-power-of-crosstab-function-in-pandas-for-data-analysis-and-visualization-6c085c269fcd
