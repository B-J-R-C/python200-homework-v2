# Week 8 Warmup — Cloud Concepts and Landscape

## Part 1: Cloud Concepts

### Question 1 — The core economic model

Cloud computing rents capacity instead of owning it: you pay per unit consumed (per instance-hour, per GB stored, per request) and you stop paying when you turn the resource off.

Owning servers is a capital purchase. You pay the full cost up front, before you know how much you'll actually use, and you keep paying in power, cooling, rack space, replacement hardware and the staff to run it — whether the machine sits at 5% utilization or 95%. Because you can't add hardware quickly, you have to buy for your peak load and then eat the idle capacity the rest of the time.

The cloud flips that from a capital expense to an operating expense and moves the risk of guessing wrong onto the provider. You provision for what you need right now and change it in minutes. The trade is that the per-hour rate is higher than the amortized cost of hardware you own — you're paying a premium for elasticity and for not having to run a data center.

### Question 2 — Vertical vs horizontal scaling

**Vertical scaling (scaling up)** means making one machine bigger — more vCPUs, more RAM, a faster GPU. The workload still runs in one place; you just give it a larger box. It's simple because nothing about the application has to change, but there's a hard ceiling at the largest instance available, and resizing usually means downtime.

**Horizontal scaling (scaling out)** means adding more machines and splitting the work across them. There's effectively no ceiling, and you can add or remove capacity while running, but the workload has to be divisible and the application has to tolerate running as many independent copies.

*When I'd choose each:* I'd scale vertically for a single-node PostgreSQL database that's running out of memory — one process owns the data, so a second machine doesn't help and I just need a bigger instance. I'd scale horizontally for a stateless web API behind a load balancer, where any server can handle any request, so adding instances multiplies throughput directly.

**The three scenarios:**

- **1,000 → 100,000 users after a viral launch:** Horizontal. Web requests are independent of each other, so the fix is more instances behind a load balancer — and it's elastic, so capacity can drop back down once the spike passes rather than being bought permanently.
- **Model training that needs a faster GPU and more RAM:** Vertical. It's a single training job in one process, so splitting it across machines isn't straightforward; the direct fix is moving it to a larger GPU instance type.
- **10 → 10,000 files per run, work can be split:** Horizontal. The files are independent, so this is embarrassingly parallel — partition them across many workers and the wall-clock time drops roughly in proportion to the number of machines.

### Question 3 — Service model classification and definitions

**Classification:**

- **Gmail — SaaS.** A finished application delivered through a browser. I don't manage servers, runtimes or updates; I just log in and use it.
- **Azure Virtual Machines — IaaS.** A raw virtual machine. Azure runs the hardware and hypervisor, and everything above that — OS, patches, runtime, application — is mine.
- **AWS S3 — IaaS.** Object storage is a primitive infrastructure building block, exposed through an API rather than an application. It's a place to put bytes, not a platform to run code on.
- **GitHub Codespaces — PaaS.** A managed development environment. I bring a repo and a container config; GitHub provisions and runs the machine, and I never touch the underlying infrastructure.
- **Snowflake — SaaS (a managed data platform).** I get a working data warehouse through a web console and a SQL endpoint with no clusters to size or patch. It's sold as a finished product, not as infrastructure I assemble.
- **Supabase — BaaS.** Backend-as-a-service: a hosted Postgres database plus auth, storage and auto-generated REST APIs, all behind one SDK, so a client app gets a complete backend without me writing or deploying a server.

**Definitions in my own words:**

**IaaS** is renting the raw building blocks — compute, storage and networking — and assembling them yourself. *Example: Azure Virtual Machines.* The provider handles the physical data center, hardware and virtualization layer. I'm responsible for everything above the hypervisor: choosing and patching the OS, installing runtimes, configuring the network and firewall rules, deploying my application, and setting up scaling, monitoring and backups myself. Maximum control, maximum work.

**PaaS** is renting a managed place to run code — as the lesson puts it, the provider manages the infrastructure but you bring your own code. *Example: Azure App Service (or GitHub Codespaces from the list above).* The provider supplies and maintains the machine, the OS and the runtime, and usually handles scaling and patching. I'm responsible for my application code and its configuration — dependencies, environment variables, the build — and not much else. Less control over the environment, far less operational work.

**SaaS** is renting the finished application. *Example: Gmail.* The provider runs absolutely all of it. I'm responsible only for my own data, my users and the settings exposed in the UI. No infrastructure decisions at all, and correspondingly no ability to change how it works.

### Question 4 — Managed data platforms

A managed data platform like Databricks, Snowflake or Dataiku is a curated layer that sits above the major cloud providers. Instead of handing you raw infrastructure, it pre-wires the pieces — storage, a query and compute engine, a SQL and notebook interface, governance, job scheduling — and optimizes the whole assembly specifically for data and analytics workloads. It generally runs on top of AWS, Azure or GCP, so it's a layer of software and operations sold above the hyperscaler rather than a competitor to it.

Using AWS or GCP directly means assembling equivalent pieces yourself: object storage, plus a compute or warehouse engine, plus a catalog, plus IAM, plus an orchestrator, and then wiring them together and keeping them running.

**What you gain:** speed of deployment. Because the pieces are already wired together and tuned for analytics, a team can be querying data the same day instead of spending weeks assembling storage, compute, a catalog and IAM into a working platform. You also get automatic scaling and tuning, one consistent governance model across your data, and no need for a dedicated platform engineering team.

**What you give up:** flexibility, and cost. The curation that makes it fast also constrains you — you work the way the platform expects, and fine-grained tuning that's available on raw infrastructure often isn't. You pay the vendor's markup on top of the underlying cloud resources, since Snowflake credits and Databricks DBUs sit above the compute they consume, so at high, steady volume it's meaningfully more expensive than running it yourself. And you take on real lock-in: proprietary storage formats, a specific SQL dialect and platform-specific pipeline definitions all make migrating away expensive once you have years of data and hundreds of jobs.

### Question 5 — When the cloud is probably not the right choice

When the dataset is small enough to fit comfortably on a single machine and there's no heavy compute demand. In that case processing locally is usually both faster and cheaper — there's no upload step, no provisioning, and nothing to pay for.

The lesson makes the same point about early prototyping. When you're still figuring out what you're building, local is the right default: the learning curve of a cloud platform is steep, and the setup overhead of getting an environment configured can easily cost more time than the actual analysis would take. The cloud earns its complexity when the data outgrows one machine or the compute demand justifies it — not before.

## Part 2: Cloud Landscape

### Question 1 — The three hyperscalers

**Amazon Web Services (AWS).** The oldest and largest of the three, with over a third of the cloud market and the broadest service catalog of any provider — EC2, S3, RDS and Lambda are the workhorses. Most likely used by organizations that want maximum breadth and maturity, from startups through large enterprises, and by anyone who wants the managed service they need next to already exist.

**Google Cloud Platform (GCP).** Strongest in data and machine learning, which follows from Google's foundational work on distributed systems — BigQuery, Vertex AI and Cloud Run are the headline products. Most likely used by organizations doing large-scale analytics or ML, where those specific tools are the reason to be there.

**Microsoft Azure.** The dominant provider in enterprise and government settings, on the strength of its integration with Windows, Active Directory and Microsoft 365 — Azure VMs, Blob Storage, Cosmos DB and Azure OpenAI are the core services. Most likely used by large enterprises and government agencies whose identity and licensing are already centralized on Microsoft.

### Question 2 — Why this course switched from Azure to Supabase

1. **Access.** Azure requires organizational provisioning — you have to join a tenant and wait for an invitation before you can do anything. Supabase accounts provision themselves in under two minutes, and the free tier is enough for the whole course, so nobody is blocked waiting on someone else.
2. **Pedagogical fit.** Azure Blob Storage treats data as opaque files sitting at a path, whereas Supabase stores rows and columns in a relational database. Querying and filtering structured data is the more transferable skill for the data roles this course is preparing people for.
3. **Pipeline coherence.** The raw and enriched zones of the ETL pipeline map cleanly onto two Supabase tables with an explicit relationship between them, which makes each stage of the pipeline easy to inspect and verify as you build it.

**My reflection:** what this suggests is that the right way to evaluate a cloud tool at the start of a project is by fit to the actual problem, not by size of the feature catalog. All three reasons are about friction rather than capability — how fast can I get access, does the tool's data model match the shape of my work, and does its structure make my project easier to reason about. Azure can do everything Supabase can and far more; that wasn't the deciding factor. The questions worth asking up front are how long until I have something running, whether the abstraction matches the problem I actually have, and whether what I learn transfers if I move. Pick the smallest tool that genuinely fits, and reach for a hyperscaler when there's a specific reason to.

### Question 3 — Matching scenarios to service categories

1. **Store 10 TB of images, retrieve by filename from anywhere** → **Object storage**. Example: **Amazon S3** (GCP Cloud Storage or Azure Blob Storage would do the same job). Flat key-value access by name over HTTP, priced per GB, effectively unlimited capacity — exactly the shape of this problem.
2. **Run an ML training job on a GPU for four hours, then shut it down** → **Compute (IaaS virtual machines)**, with a GPU instance type. Example: **AWS EC2** — a p3 or g5 instance — or GCP Compute Engine with a GPU attached. Billed by the hour, so terminating the instance stops the cost.
3. **Host a web API that scales up on traffic spikes and back down when quiet** → **Serverless compute**. Example: **AWS Lambda** (GCP Cloud Functions or Azure Functions are the equivalents). Scaling is automatic and you pay per invocation rather than for idle capacity.
4. **Send structured data to a large language model and get text back** → **LLM API**. Example: **AWS Bedrock** (GCP Vertex AI or Azure OpenAI are the equivalents). A hosted model behind an HTTP endpoint, priced per token.

### Question 4 — A multi-provider stack

**The project:** a daily running-weather tracker. Every morning it pulls the previous day's weather for my city from a public API, stores it, asks an LLM whether conditions were good for running and why, and shows the last 30 days on a small web dashboard.

**A plausible stack:**

| Layer | Service category | Product |
|---|---|---|
| Weather rows, raw and enriched | Managed relational DB | **Supabase** |
| Daily ingest job on a schedule | Serverless compute | **AWS Lambda** |
| Enrichment — the running verdict | LLM API | **AWS Bedrock** |
| Archived raw API responses | Object storage | **Amazon S3** |

Three vendors across four layers, which is fairly typical — each piece is the easiest thing for its particular job, and none of them is hard to swap out.

**Is there a benefit to consolidating?** Yes, several. One bill and one vendor relationship instead of four. A single identity and permissions model rather than four sets of API keys to rotate and keep out of Git. Components in the same provider talk to each other over the internal network, which is faster and avoids egress charges. And there's less context-switching — one console, one set of docs, one support channel — plus the possibility of volume discounts as usage grows.

**What I'd give up:** best-of-breed fit at every layer. Consolidating onto AWS would mean trading Supabase for RDS, and for a project this small that's a real downgrade in setup time — RDS wants a VPC, subnet groups and security rules before it will accept a connection, where Supabase gives me a queryable table and an API immediately. I'd also be limited to whichever models Bedrock carries rather than picking the one I actually want. And I'd lose negotiating leverage while increasing the blast radius: a single outage, price change or account problem would take out the whole project instead of one layer of it. For something this size the multi-provider version is simpler in practice, even though it looks more fragmented on paper.