"""
FintechCID -- LangGraph Orchestration Graph
===============================================
Compiles the StateGraph with:
  - MemorySaver checkpointer for HITL persistence
  - interrupt_before=["compliance_arbitrator"] for human review breakpoint

Node map
--------
screen_transaction
    LOW  --> END
    HIGH --> document_forensics
                --> [BREAKPOINT] compliance_arbitrator
                        --> END
"""

from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from agents.state import AgentState
from agents.transaction_screener import screen_transaction
from agents.document_forensics import document_forensics_node
from agents.compliance_arbitrator import compliance_arbitrator_node


# ---------------------------------------------------------------------------
# Routing logic
# ---------------------------------------------------------------------------

def _route_after_screener(state: AgentState) -> str:
    if state.get("suspicion_level") == "HIGH":
        return "document_forensics"
    return END


# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------

def build_graph(with_hitl: bool = True):
    """
    Compile and return the LangGraph StateGraph.

    Parameters
    ----------
    with_hitl : bool
        If True (default), compile with MemorySaver + interrupt_before the
        compliance_arbitrator node for human-in-the-loop review.
        Set False for automated testing without breakpoints.
    """
    g = StateGraph(AgentState)

    # Nodes
    g.add_node("screen_transaction",    screen_transaction)
    g.add_node("document_forensics",    document_forensics_node)
    g.add_node("compliance_arbitrator", compliance_arbitrator_node)

    # Entry point
    g.set_entry_point("screen_transaction")

    # Screener routing
    g.add_conditional_edges(
        "screen_transaction",
        _route_after_screener,
        {
            "document_forensics": "document_forensics",
            END: END,
        },
    )

    # Forensics always feeds into compliance arbitrator
    g.add_edge("document_forensics", "compliance_arbitrator")

    # Compliance arbitrator terminates graph
    g.add_edge("compliance_arbitrator", END)

    if with_hitl:
        checkpointer = MemorySaver()
        return g.compile(
            checkpointer=checkpointer,
            interrupt_before=["compliance_arbitrator"],
        )
    else:
        return g.compile()
