import json
import argparse
import os
from collections import defaultdict
from typing import Dict, Set, Tuple, Any, List, Union

class ExtraInfoRegistry:
    def __init__(self):
        self.registry: List[Dict[str, Any]] = []
        self.hash_map: Dict[str, int] = {}

    def get_id(self, extra_info: Dict[str, Any]) -> int:
        """Returns a unique integer ID for a given extra_info dictionary."""
        info_str = json.dumps(extra_info, sort_keys=True)
        if info_str not in self.hash_map:
            idx = len(self.registry)
            self.registry.append(extra_info)
            self.hash_map[info_str] = idx
        return self.hash_map[info_str]

def parse_args():
    parser = argparse.ArgumentParser(description="Rank DuckDB operators by their frequency across different plans and total timing.")
    parser.add_argument("directory", help="Path to the directory containing JSON plans (*_table.json and *_view.json)")
    parser.add_argument("--top", type=int, default=20, help="Number of top operators to display (default: 20)")
    parser.add_argument("--extra-info", action="store_true", help="Include full extra_info in uniqueness criteria")
    return parser.parse_args()

def get_operator_signature(node: Dict[str, Any], include_extra: bool, registry: ExtraInfoRegistry) -> Tuple | None:
    """
    Returns a unique signature for an operator.
    Handles 'operator_name', 'operator_type', and 'name' (for view plans).
    """
    op_name = node.get("operator_name") or node.get("operator_type") or node.get("name")
    if not op_name:
        return None
    
    extra_info = node.get("extra_info", {})
    est_cardinality = str(extra_info.get("Estimated Cardinality", "0"))
    
    if include_extra:
        extra_id = registry.get_id(extra_info)
        return (op_name, est_cardinality, extra_id)
    
    return (op_name, est_cardinality)

def collect_stats(data: Union[Dict, List], 
                  file_signatures: Set[Tuple], 
                  total_timings: Dict[Tuple, float], 
                  include_extra: bool, 
                  registry: ExtraInfoRegistry,
                  is_analyze: bool):
    """Recursively collect unique signatures per file and cumulative timings (if analyze)."""
    if isinstance(data, list):
        for item in data:
            collect_stats(item, file_signatures, total_timings, include_extra, registry, is_analyze)
        return

    sig = get_operator_signature(data, include_extra, registry)
    if sig:
        file_signatures.add(sig)
        if is_analyze:
            timing = float(data.get("operator_timing", 0.0))
            total_timings[sig] += timing

    # Recurse into children
    children = data.get("children", [])
    if children:
        collect_stats(children, file_signatures, total_timings, include_extra, registry, is_analyze)

def main():
    args = parse_args()
    
    if not os.path.isdir(args.directory):
        print(f"Error: {args.directory} is not a directory.")
        return

    registry = ExtraInfoRegistry()
    # Map of signature Tuple -> count of unique files
    sig_table_counts = defaultdict(int)
    sig_view_counts = defaultdict(int)
    # Map of signature Tuple -> total cumulative timing (from table plans)
    sig_total_timing = defaultdict(float)
    
    all_files = os.listdir(args.directory)
    table_files = [f for f in all_files if f.endswith("_table.json")]
    view_files = [f for f in all_files if f.endswith("_view.json")]
    
    if not table_files and not view_files:
        print(f"No *_table.json or *_view.json files found in {args.directory}")
        return

    # Process Table Plans (EXPLAIN ANALYZE)
    for filename in table_files:
        filepath = os.path.join(args.directory, filename)
        try:
            with open(filepath, 'r') as f:
                plan_data = json.load(f)
                file_signatures: Set[Tuple] = set()
                collect_stats(plan_data, file_signatures, sig_total_timing, args.extra_info, registry, is_analyze=True)
                for sig in file_signatures:
                    sig_table_counts[sig] += 1
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not parse {filename}: {e}")

    # Process View Plans (EXPLAIN)
    for filename in view_files:
        filepath = os.path.join(args.directory, filename)
        try:
            with open(filepath, 'r') as f:
                plan_data = json.load(f)
                file_signatures: Set[Tuple] = set()
                # View plans don't have timing, so we pass a dummy dict
                collect_stats(plan_data, file_signatures, {}, args.extra_info, registry, is_analyze=False)
                for sig in file_signatures:
                    sig_view_counts[sig] += 1
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not parse {filename}: {e}")

    # All unique signatures across both sets
    all_signatures = set(sig_table_counts.keys()) | set(sig_view_counts.keys())

    # Convert to list for ranking
    ranking = []
    for sig in all_signatures:
        entry = {
            "operator": sig[0],
            "est_cardinality": sig[1],
            "table_occurrences": sig_table_counts[sig],
            "view_occurrences": sig_view_counts[sig],
            "total_timing": sig_total_timing[sig]
        }
        if args.extra_info:
            entry["extra_info_id"] = sig[2]
        ranking.append(entry)

    # Rank by number of table occurrences (descending), then total timing (descending)
    ranking.sort(key=lambda x: (-x["table_occurrences"], -x["total_timing"], -x["view_occurrences"], x["operator"]))

    # Determine column widths
    col_op = 35
    col_card = 20
    col_t_occ = 12
    col_v_occ = 12
    col_time = 18
    col_extra = 12 if args.extra_info else 0
    
    total_width = col_op + col_card + col_t_occ + col_v_occ + col_time + col_extra + 5
    
    print("\n" + "="*total_width)
    header = (f"{'Operator':<{col_op}} {'Est. Card.':<{col_card}} {'Table Occ':>{col_t_occ}} "
              f"{'View Occ':>{col_v_occ}} {'Total Timing (s)':>{col_time}}")
    if args.extra_info:
        header += f" {'Extra ID':>{col_extra}}"
    print(header)
    print("-" * total_width)
    
    display_list = ranking[:args.top]
    for entry in display_list:
        line = (f"{entry['operator']:<{col_op}} {entry['est_cardinality']:<{col_card}} "
                f"{entry['table_occurrences']:>{col_t_occ}} {entry['view_occurrences']:>{col_v_occ}} "
                f"{entry['total_timing']:>{col_time}.6f}")
        if args.extra_info:
            line += f" {entry['extra_info_id']:>{col_extra}}"
        print(line)
    
    if len(ranking) > args.top:
        print(f"\n... (Showing top {args.top} out of {len(ranking)} unique operators)")
    print("="*total_width + "\n")

if __name__ == "__main__":
    main()
