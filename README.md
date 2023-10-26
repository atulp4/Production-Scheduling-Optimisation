# Production-Scheduling-Optimisation-and-Automation

The project was an intensive and gruesome undertaking during my first year in the field of Data Science. The project revolved around automating 
the manual Cleaning In Place scheduling of a manufacturing plant in Reiti, Italy. Closely working with business heads at the plant, helped 
understand the task at hand. In simple words, there were 50+ manufacturing machines in the plant, that were connected to 4 cleaning machines 
through pipes. The manufacturing machines had pre-planned production scedules according to which their cleaning schedules had to be 
scheduled. There were different constraints at hand that had to be considered. To successfully do this, a model was developed that could 
generate schedules for all pre-planned production activities. A front-end was also developed and deployed that helped the Head Scheduler
of the plant generate schedules and monitor KPI's.

## Scheduling Model code

This Jupyter notebook contains the code to generate an optimized production schedule for pharmaceutical manufacturing processes. The scheduling code is designed to optimize the cleaning and maintenance schedule for pharmaceutical manufacturing equipment based on various constraints.

           - Scheduling_Model.ipynb

It first imports the necessary Python libraries like pandas, numpy, datetime, etc. 

Then it defines several helper functions to:

- Handle date constraints when assigning cleaning schedules 
- Get available CIP groups and constraints for a resource
- Calculate constraints like DHT and CHT
- Assign CIP groups and constraints to a cleaning
- Allocate cleaning to utilizations with available slots
- Check for overlaps and violations in assigned cleaning schedules
- Include recleaning where needed
- Push cleaning schedules earlier to optimize schedules

It first reads in the raw utilization data, mapping parameters, and any pre-scheduled cleaning data. It does preprocessing on the data like handling date formats, missing values, etc.

Some key steps:

- It calculates metrics like idle time between usages, last possible cleaning time, etc for each equipment. This is used later for assigning cleaning slots.
- It determines cleaning constraints like maximum Allowable Downtime (DHT) and Allowable Clean Hold Time (CHT) for each equipment from the mapping file.
- It checks for any pre-scheduled frozen cleaning that must be incorporated into the schedule. It assigns these as Post Utilization Cleaning or Pre Cleaning.
- It assigns initial cleaning slots for each equipment utilization trying to satisfy all constraints. It does this sequentially equipment by equipment.
- For high priority equipments, it adjusts the cleaning slot to start earlier if possible to avoid CHT violations.
- It looks for any CHT violations and tries to push the cleaning slots earlier in time to avoid it if possible. This is done sequentially pushing one cleaning at a time.
- It incorporates any required re-cleanings and pre-cleanings to avoid CHT violations.
- It handles scheduled maintenance activities by assigning cleaning slots before/after maintenances if enough idle time is available.
- Finally it outputs the optimized cleaning schedule satisfying all constraints.

The core optimization occurs by iteratively assigning cleaning schedules into available CIP/constraint slots and pushing schedules earlier until an optimal schedule is reached.

## Quality Check code

This code runs quality checks on the generated optimized schedule to validate it meets requirements.

           - QC_Quality_Check.py

It has functions to check for:

- Missed cleanings 
- Overlapping cleaning schedules
- DHT violations
- CHT violations  
- Correct mapping of flexible resources
- Valid parallel cleaning mappings
- Count of pre-cleanings vs recleaning

It generates a report dataframe for each check, concantenates them, and exports to an Excel file. This allows data scientists and engineers to easily validate if the optimized schedule meets all constraints and quality criteria.

## Business Metrics code

This code calculates business metrics from the optimized schedule to evaluate the utilization of resources. 

           - Business_metrics.py

It computes metrics like:

- Total cleanings per resource 
- Cleaning time utilization
- Idle time percentage
- Usage percentage
- CIP group usage percentage
- Inter-arrival cleaning times

These metrics quantify the utilization and performance of the schedule and help identify improvement opportunities. The metrics are calculated using aggregations on the schedule dataframe and exported to an Excel report. This provides clear visibility into how well resources are utilized in the optimized schedule.

In summary, these scripts work together to generate an optimized schedule, validate it meets requirements, and calculate metrics to evaluate performance. The Scheduling_Model handles core optimization, the QC code validates quality, and the Business_metrics calculates utilization metrics for the final schedule output.


For deeper understanding check - 

1. Contextualised problem definition

           - Problem Definition_Takeda.pdf

2. Understanding mapping of pipes from manufacturing machines to cleaning machines

           - Mapping.pdf

3. Understanding all metrics involved

           - Metrics_Dictionary.pdf

4. Developing a model that successfully generates schedules using greedy algorithm, recursion and back tracking

           - Scheduling_Model.ipynb

5. Check the results for its viability and correctness by making a checklist, automated script for Quality Check and result
   
           - QC_Checklist.pdf
           - QC_Quality_Check.py
           - QC_Result.pdf
   
6. Generate business metrics by developing automated scripts
   
           - Business_metrics.py
           - Business_metrics_result.pdf

7. Creating a front-end and deploying it
    
           - Front_End_Code.py
           - Front_End_Snippet.pdf


