"""
Project Planning AI Agent - Multi-agent prototype (Python)

This single-file prototype demonstrates a minimal multi-agent architecture for
project planning. It includes:
 - Architecture ASCII diagram + explanation
 - Agent classes: PlannerAgent, SchedulerAgent, RiskAgent, ReporterAgent
 - Tool wrappers (placeholders) for Jira and Calendar
 - Simple orchestration loop
 - Example run with a sample project

NOTES:
 - Replace `call_llm` with your preferred LLM client (OpenAI, Anthropic, etc.)
 - This is a scaffold intended for quick iteration; production use should add
   error handling, retry policies, authentication secrets management, async
   execution, and robust tool integrations.

Run: python project_planning_ai_agent_prototype.py

"""

# ------------------------ Architecture (ASCII) -----------------------------
#
#                           +------------------+
#                           |   User / PM UI   |
#                           +--------+---------+
#                                    |
#                                    v
#                          +----------------------+
#                          | Orchestrator / Agent |
#                          |    Coordinator       |
#                          +---+---+---+---+------+    <- manages turn-taking
#                              |   |   |   |
#                 +------------+   |   |   +------------+
#                 |                |   |                |
#                 v                v   v                v
#         +--------------+  +--------------+  +--------------+
#         | PlannerAgent |  | SchedulerAgent|  | RiskAgent    |
#         +--------------+  +--------------+  +--------------+
#                 |                |                 |
#                 |                |                 |
#                 +-------+--------+-------+---------+
#                         |                |
#                         v                v
#                    +----------+      +-----------+
#                    | Jira API |      | Calendar  |
#                    +----------+      +-----------+
#
#  ReporterAgent collects decisions and creates a human-readable plan:
#  - Tasks, owners, estimates, milestones, risks, and Jira issues created.
#
# ------------------------ Requirements ------------------------------------
# - Python 3.9+
# - Install any LLM client (openai, anthropic, etc.) if you want to connect
#   to a real model. This scaffold uses a single `call_llm` function you should
#   implement for your provider.
# - Add API clients for Jira / Calendar as needed (jira, google-api-python-client)
# --------------------------------------------------------------------------

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import time
import uuid
import pprint

pp = pprint.PrettyPrinter(indent=2)

# ------------------------ Utilities ---------------------------------------

def call_llm(prompt: str, temperature: float = 0.2, max_tokens: int = 512) -> str:
    """
    Replace this function with a real LLM call. Example (OpenAI):

    import openai
    openai.api_key = os.getenv('OPENAI_API_KEY')
    resp = openai.ChatCompletion.create(
      model='gpt-5-thinking-mini',
      messages=[{'role':'user','content': prompt}],
      temperature=temperature,
      max_tokens=max_tokens
    )
    return resp['choices'][0]['message']['content']

    For now this function returns deterministic mock answers to allow the
    prototype to run offline.
    """
    # Basic mock behavior based on keywords
    if 'break down' in prompt.lower() or 'decompose' in prompt.lower():
        return (
            "1) Requirement analysis (2 days)\n"
            "2) Design & Interfaces (3 days)\n"
            "3) Driver implementation (5 days)\n"
            "4) Integration & Testing (4 days)\n"
            "Assign: Senior kernel engineer (Driver), QA engineer (Integration)"
        )
    if 'schedule' in prompt.lower() or 'estimate' in prompt.lower():
        return (
            "Milestone A: Week 1-1.5\n"
            "Milestone B: Week 2-3\n"
            "Critical path: Design -> Driver implementation -> Integration"
        )
    if 'risks' in prompt.lower() or 'identify' in prompt.lower():
        return (
            "1) Upstream API changes could break build (medium)\n"
            "2) Lack of test hardware (high)\n"
            "3) DTS conflicts (low)\n"
        )
    if 'report' in prompt.lower() or 'summary' in prompt.lower():
        return "Generated plan: 4 milestones, 3 owners, estimated 14 working days."
    # fallback
    return "[LLM] I'm a mocked model. Replace call_llm with real LLM integration."

# ------------------------ Data Models -------------------------------------

@dataclass
class Task:
    id: str
    title: str
    description: str
    estimate_days: float
    owner: Optional[str] = None
    depends_on: List[str] = field(default_factory=list)

@dataclass
class ProjectPlan:
    project_id: str
    title: str
    tasks: List[Task] = field(default_factory=list)
    milestones: Dict[str, List[str]] = field(default_factory=dict)
    risks: List[str] = field(default_factory=list)

# ------------------------ Tool Wrappers (placeholders) ---------------------

class JiraTool:
    def __init__(self, base_url: str = 'https://your-jira.example'):
        self.base_url = base_url

    def create_issue(self, summary: str, description: str, assignee: Optional[str] = None) -> Dict[str, Any]:
        # Replace with actual Jira API calls (requests or jira client)
        issue_key = f"PROJ-{int(time.time()) % 10000}"
        print(f"[Jira] Created issue {issue_key}: {summary} (assignee={assignee})")
        return {"key": issue_key, "summary": summary}

class CalendarTool:
    def schedule_meeting(self, title: str, when: str, attendees: List[str]):
        # Replace with Google Calendar / Outlook API call
        print(f"[Calendar] Scheduled '{title}' on {when} for {len(attendees)} attendees")
        return {"event_id": str(uuid.uuid4()), "title": title, "when": when}

# ------------------------ Agents ------------------------------------------

class BaseAgent:
    def __init__(self, name: str):
        self.name = name

    def act(self, context: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

class PlannerAgent(BaseAgent):
    def __init__(self):
        super().__init__('PlannerAgent')

    def act(self, context: Dict[str, Any]) -> Dict[str, Any]:
        prompt = (
            f"You are PlannerAgent. Break down the project into tasks.\n"
            f"Project title: {context['title']}\n"
            f"Description: {context.get('description','')}\n"
            "Return a numbered list with estimated days and suggested owners."
        )
        print(f"[{self.name}] calling LLM to decompose project...")
        resp = call_llm(prompt)
        print(f"[{self.name}] LLM response:\n{resp}\n")

        # Simple parser for mocked LLM output - in real code, use JSON from LLM
        lines = [l.strip() for l in resp.splitlines() if l.strip()]
        tasks = []
        for i, line in enumerate(lines, start=1):
            # naive parsing
            title = line.split(':')[0][:80]
            estimate = 1.0
            owner = None
            tasks.append(Task(id=f"T{i}", title=title, description=line, estimate_days=estimate, owner=owner))

        return {"tasks": tasks}

class SchedulerAgent(BaseAgent):
    def __init__(self):
        super().__init__('SchedulerAgent')

    def act(self, context: Dict[str, Any]) -> Dict[str, Any]:
        # context should contain tasks
        prompt = (
            f"You are SchedulerAgent. Given tasks, estimate milestones and critical path.\n"
            f"Tasks: {[t.title for t in context['tasks']]}\n"
            "Return milestones and suggested task ordering."
        )
        print(f"[{self.name}] calling LLM to schedule tasks...")
        resp = call_llm(prompt)
        print(f"[{self.name}] LLM response:\n{resp}\n")
        # parse minimal
        milestones = {"M1": [t.id for t in context['tasks'][:2]], "M2": [t.id for t in context['tasks'][2:]]}
        ordering = [t.id for t in context['tasks']]
        return {"milestones": milestones, "ordering": ordering}

class RiskAgent(BaseAgent):
    def __init__(self):
        super().__init__('RiskAgent')

    def act(self, context: Dict[str, Any]) -> Dict[str, Any]:
        prompt = (
            f"You are RiskAgent. Identify top risks for project: {context['title']}.\n"
            f"Tasks: {[t.title for t in context['tasks']]}\n"
            "Return top risks with short mitigation notes."
        )
        print(f"[{self.name}] calling LLM to identify risks...")
        resp = call_llm(prompt)
        print(f"[{self.name}] LLM response:\n{resp}\n")
        risks = [r.strip() for r in resp.splitlines() if r.strip()]
        return {"risks": risks}

class ReporterAgent(BaseAgent):
    def __init__(self, jira_tool: JiraTool, calendar_tool: CalendarTool):
        super().__init__('ReporterAgent')
        self.jira = jira_tool
        self.calendar = calendar_tool

    def act(self, context: Dict[str, Any]) -> Dict[str, Any]:
        prompt = (
            f"You are ReporterAgent. Summarize plan and create Jira issues for tasks.\n"
            f"Project: {context['title']}\n"
        )
        print(f"[{self.name}] calling LLM to prepare report...")
        summary = call_llm(prompt)
        print(f"[{self.name}] LLM response:\n{summary}\n")

        created = []
        for t in context['tasks']:
            issue = self.jira.create_issue(summary=t.title, description=t.description, assignee=t.owner)
            created.append(issue)

        # optional: schedule kickoff meeting
        event = self.calendar.schedule_meeting(title=f"Kickoff: {context['title']}", when="2025-11-05T10:00:00+05:30", attendees=[t.owner for t in context['tasks'] if t.owner])

        return {"summary": summary, "issues": created, "kickoff_event": event}

# ------------------------ Orchestrator ------------------------------------

class Orchestrator:
    def __init__(self):
        self.planner = PlannerAgent()
        self.scheduler = SchedulerAgent()
        self.risk = RiskAgent()
        self.jira = JiraTool()
        self.calendar = CalendarTool()
        self.reporter = ReporterAgent(self.jira, self.calendar)

    def run(self, title: str, description: str) -> ProjectPlan:
        context = {"title": title, "description": description}

        # Step 1: Planner decomposes
        p_out = self.planner.act(context)
        tasks: List[Task] = p_out['tasks']
        context['tasks'] = tasks

        # Step 2: Scheduler orders and creates milestones
        s_out = self.scheduler.act(context)
        context.update(s_out)

        # Step 3: Risk analysis
        r_out = self.risk.act(context)
        context.update(r_out)

        # Step 4: Reporter creates issues and summary
        rep_out = self.reporter.act(context)
        context.update(rep_out)

        # Build ProjectPlan
        plan = ProjectPlan(project_id=str(uuid.uuid4()), title=title)
        plan.tasks = tasks
        plan.milestones = context.get('milestones', {})
        plan.risks = context.get('risks', [])

        return plan

# ------------------------ Example usage ----------------------------------

if __name__ == '__main__':
    orchestrator = Orchestrator()

    sample_title = 'Port SPI and NAND drivers to Linux 6.6'
    sample_desc = (
        'Port existing SPI and NAND drivers from legacy kernel to Linux 6.6.\n'
        'Deliver: buildable kernel tree, device tree updates, driver patches, and validation tests.'
    )

    print('\n=== Running multi-agent prototype for sample project ===\n')
    plan = orchestrator.run(sample_title, sample_desc)

    print('\n=== Generated Project Plan ===')
    pp.pprint(plan)

    print('\n=== Tasks (detailed) ===')
    for t in plan.tasks:
        print(f"- {t.id}: {t.title} (est {t.estimate_days}d) owner={t.owner}")

    print('\n=== Milestones ===')
    pp.pprint(plan.milestones)

    print('\n=== Risks ===')
    pp.pprint(plan.risks)

    print('\nPrototype complete. Next steps: connect a real LLM, improve parsing, add auth.')

