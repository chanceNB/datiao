from __future__ import annotations
from collections import defaultdict
from .models import AlignmentRecord, LABEL_ORDER

def align_truth(episodes, truth_by_case):
    by_case=defaultdict(list)
    for ep in episodes: by_case[ep.case_id].append(ep)
    labels={ep.episode_id:set() for ep in episodes}; records=[]
    for case_id, events in truth_by_case.items():
        for event in events:
            candidates=[ep for ep in by_case.get(case_id,[]) if set(ep.point_refs)&set(event.source_point_ids)]
            if len(candidates)==0: status='UNMATCHED'
            elif len(candidates)==1: status='ONE_TO_ONE'
            else: status='ONE_TO_MANY'
            records.append(AlignmentRecord(case_id=case_id, truth_event_id=event.event_id, event_type=event.event_type,
                matched_episode_ids=tuple(ep.episode_id for ep in candidates), alignment_status=status))
            if len(candidates)==1: labels[candidates[0].episode_id].add(event.event_type)
    return labels, tuple(records)

def target_vector(labels):
    return tuple(1 if name in labels else 0 for name in LABEL_ORDER)
