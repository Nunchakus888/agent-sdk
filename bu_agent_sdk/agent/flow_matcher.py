"""
Flow Matcher - Keyword matching for flows.

Design principles:
1. Single responsibility - FlowMatcher only handles matching, not execution
2. Execution is delegated to trigger_flow tool (defined in config)
3. Keyword matching is fast, deterministic, zero LLM cost
"""

import re
import logging
from dataclasses import dataclass, field

from bu_agent_sdk.schemas import FlowDefinition, FlowType, MatchType

logger = logging.getLogger("agent_sdk.flow_matcher")


@dataclass
class FlowMatchResult:
    """Flow match result."""
    matched: bool = False
    flow: FlowDefinition | None = None
    matched_pattern: str | None = None


class KeywordMatcher:
    """Keyword matcher - supports exact/contains/regex modes."""

    @staticmethod
    def match(user_message: str, flow: FlowDefinition) -> str | None:
        """
        Check if user message matches flow's trigger_patterns.

        Returns:
            Matched pattern string if matched, None otherwise.
        """
        if not flow.trigger_patterns:
            return None

        user_msg_lower = user_message.lower().strip()

        for pattern in flow.trigger_patterns:
            pattern_lower = pattern.lower().strip()

            match flow.match_type:
                case MatchType.EXACT:
                    if user_msg_lower == pattern_lower:
                        return pattern

                case MatchType.CONTAINS:
                    if pattern_lower in user_msg_lower:
                        return pattern

                case MatchType.REGEX:
                    try:
                        if re.search(pattern, user_message, re.IGNORECASE):
                            return pattern
                    except re.error as e:
                        logger.warning(f"Invalid regex pattern '{pattern}': {e}")

        return None


@dataclass
class FlowMatcher:
    """
    Flow matcher - keyword matching only.

    Execution is delegated to trigger_flow tool.

    Usage:
        matcher = FlowMatcher(flows=config.flows)

        # Try keyword matching
        result = matcher.match_keyword(user_message)
        if result.matched:
            # Call trigger_flow tool with flow_id
            await trigger_flow_tool.execute(flow_id=result.flow.flow_id)
    """

    flows: list[FlowDefinition]

    # Internal cache
    _keyword_flows: list[FlowDefinition] = field(default_factory=list, repr=False)
    _intent_flows: list[FlowDefinition] = field(default_factory=list, repr=False)

    def __post_init__(self):
        """Classify flows by type."""
        for flow in self.flows:
            if flow.type == FlowType.KEYWORD:
                self._keyword_flows.append(flow)
            else:
                self._intent_flows.append(flow)

        logger.debug(
            f"FlowMatcher: {len(self._keyword_flows)} keyword, "
            f"{len(self._intent_flows)} intent flows"
        )

    def match_keyword(self, user_message: str) -> FlowMatchResult:
        """Try keyword matching."""
        for flow in self._keyword_flows:
            matched_pattern = KeywordMatcher.match(user_message, flow)
            if matched_pattern:
                logger.info(f"Keyword matched: {flow.flow_id}, pattern={matched_pattern}")
                return FlowMatchResult(
                    matched=True,
                    flow=flow,
                    matched_pattern=matched_pattern,
                )
        return FlowMatchResult(matched=False)

    @property
    def keyword_flows(self) -> list[FlowDefinition]:
        return self._keyword_flows

    @property
    def intent_flows(self) -> list[FlowDefinition]:
        return self._intent_flows

    @property
    def has_keyword_flows(self) -> bool:
        return len(self._keyword_flows) > 0
