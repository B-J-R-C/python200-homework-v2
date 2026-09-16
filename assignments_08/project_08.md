# Video Link
https://drive.google.com/file/d/1bhXuKcnmiaGc8yV6IGywKntePZ0pyiCj/view?usp=sharing 

# Part A: Supabase Setup Confirmation
My Supabase project is successfully set up. I have my `SUPABASE_URL` and `SUPABASE_KEY` saved in a local `.env` file (which is safely added to my `.gitignore`), and both the `weather_raw` and `weather_enriched` tables are created with Row Level Security disabled.

# Part B: Cloud Cost Analysis

**Scenario A (Lightweight compute):** 
A `t3.micro` EC2 instance running 160 hours a month in US East (N. Virginia) costs approximately **$1.66 / month**. I was surprised at how incredibly cheap it is to run lightweight compute for a standard workweek schedule.

**Scenario B (Heavy analytics workload):** 
This workload costs roughly **$2,393.13 / month**. The `p3.2xlarge` GPU instance running 24/7 is the bulk of the cost at ~$2,233.80, combined with a `db.m5.large` RDS database at ~$136.33, and 1TB of S3 storage for $23.00. The sheer cost of leaving a GPU instance running 24/7 was eye-opening.

**Exploration Find:** 
While exploring the calculator, I checked out "Data Transfer" and found that moving 10 TB of data *out* of AWS to the internet costs nearly $900/month in egress fees, which is astronomically more expensive than the $23 it costs to actually store it in S3!

**Comparison Summary:** 
The massive gap between $1.66 and $2,393 demonstrates that GPU instances should never be left idling 24/7. It tells me that expensive GPU compute is only worth it if you can spin it up on-demand, run your specific workload, and shut it down immediately after.