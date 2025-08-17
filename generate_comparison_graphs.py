## this script will compile the statistics of the benchmark
## for each benchmark log file in the logs folder, it will read the jsonl file, process accuracy, average latency, and input/output token usage, and make 3 bar graphs comparing the results from each of the different tests

import json
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def parse_jsonl_file(file_path):
    """Parse a JSONL file and extract benchmark statistics"""
    stats = {
        'name': '',
        'total_questions': 0,
        'correct_answers': 0,
        'accuracy': 0.0,
        'latencies': [],
        'input_tokens': [],
        'output_tokens': [],
        'total_input_tokens': 0,
        'total_output_tokens': 0,
        'avg_latency': 0.0,
        'avg_input_tokens': 0.0,
        'avg_output_tokens': 0.0
    }
    
    # Extract benchmark name from filename
    filename = Path(file_path).name
    if 'gpt-4o' in filename or 'gpt4o' in filename:
        stats['name'] = 'GPT-4o'
    elif 'hybrid' in filename:
        stats['name'] = 'Equipade'
    else:
        stats['name'] = filename.replace('.jsonl', '')
    
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                try:
                    data = json.loads(line)
                    stats['total_questions'] += 1
                    
                    # Count correct answers
                    if data.get('correct', False):
                        stats['correct_answers'] += 1
                    
                    # Collect latency data
                    latency = data.get('latency_ms')
                    if latency is not None:
                        stats['latencies'].append(latency)
                    
                    # Collect token data
                    tokens = data.get('tokens', {})
                    if tokens:
                        input_tokens = tokens.get('input')
                        output_tokens = tokens.get('output')
                        
                        if input_tokens is not None:
                            stats['input_tokens'].append(input_tokens)
                            stats['total_input_tokens'] += input_tokens
                        
                        if output_tokens is not None:
                            stats['output_tokens'].append(output_tokens)
                            stats['total_output_tokens'] += output_tokens
                
                except json.JSONDecodeError:
                    continue  # Skip malformed lines
    
    # Calculate derived statistics
    if stats['total_questions'] > 0:
        stats['accuracy'] = (stats['correct_answers'] / stats['total_questions']) * 100
    
    if stats['latencies']:
        stats['avg_latency'] = sum(stats['latencies']) / len(stats['latencies'])
    
    # Calculate average tokens per query
    if stats['total_questions'] > 0:
        stats['avg_input_tokens'] = stats['total_input_tokens'] / stats['total_questions']
        stats['avg_output_tokens'] = stats['total_output_tokens'] / stats['total_questions']
    
    return stats

def find_log_files():
    """Find all JSONL log files in the logs directory"""
    logs_dir = 'logs'
    if not os.path.exists(logs_dir):
        print(f"Logs directory '{logs_dir}' not found!")
        return []
    
    # Find all .jsonl files
    pattern = os.path.join(logs_dir, '*.jsonl')
    files = glob.glob(pattern)
    
    # Filter out empty files and files with ' copy' in the name
    valid_files = []
    for file in files:
        if ' copy' not in file and os.path.getsize(file) > 0:
            valid_files.append(file)
    
    return valid_files

def create_comparison_graphs(all_stats):
    """Create three bar graphs comparing the benchmarks"""
    if not all_stats:
        print("No valid statistics found to plot!")
        return
    
    # Prepare data for plotting
    names = [stats['name'] for stats in all_stats]
    accuracies = [stats['accuracy'] for stats in all_stats]
    avg_latencies = [stats['avg_latency'] for stats in all_stats]
    avg_input_tokens = [stats['avg_input_tokens'] for stats in all_stats]
    avg_output_tokens = [stats['avg_output_tokens'] for stats in all_stats]
    
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Benchmark Comparison Results', fontsize=16, fontweight='bold')
    
    # Colors for bars
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Graph 1: Accuracy
    bars1 = ax1.bar(names, accuracies, color=colors[:len(names)])
    ax1.set_title('Accuracy (%)')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_ylim(0, 110)  # Give more space at the top for text labels
    
    # Add value labels on bars
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{acc:.1f}%', ha='center', va='bottom')
    
    # Graph 2: Average Latency
    bars2 = ax2.bar(names, avg_latencies, color=colors[:len(names)])
    ax2.set_title('Average Latency (ms)')
    ax2.set_ylabel('Latency (ms)')
    ax2.set_ylim(0, max(avg_latencies) * 1.15)  # Give more space at the top for text labels
    
    # Add value labels on bars
    for bar, lat in zip(bars2, avg_latencies):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + max(avg_latencies)*0.02,
                f'{lat:.0f}ms', ha='center', va='bottom')
    
    # Graph 3: Average Token Usage per Query (Input + Output)
    x = np.arange(len(names))
    width = 0.35
    
    bars3a = ax3.bar(x - width/2, avg_input_tokens, width, label='Avg Input Tokens', 
                     color='lightblue', alpha=0.8)
    bars3b = ax3.bar(x + width/2, avg_output_tokens, width, label='Avg Output Tokens', 
                     color='lightcoral', alpha=0.8)
    
    ax3.set_title('Average Token Usage per Query')
    ax3.set_ylabel('Average Tokens')
    ax3.set_xticks(x)
    ax3.set_xticklabels(names)
    ax3.legend()
    
    # Set y-axis limit with extra space for text labels
    max_tokens = max(max(avg_input_tokens), max(avg_output_tokens))
    ax3.set_ylim(0, max_tokens * 1.15)
    
    # Add value labels on token bars
    for bars, values in [(bars3a, avg_input_tokens), (bars3b, avg_output_tokens)]:
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + max_tokens*0.02,
                    f'{val:.0f}', ha='center', va='bottom', fontsize=9)
    
    # Rotate x-axis labels if they're long
    for ax in [ax1, ax2]:
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save the plot
    output_path = 'benchmark_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Graphs saved to: {output_path}")
    
    # Show the plot
    plt.show()

def print_summary_table(all_stats):
    """Print a summary table of all statistics"""
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    
    # Table header
    print(f"{'Benchmark':<20} {'Questions':<10} {'Correct':<8} {'Accuracy':<10} {'Avg Latency':<12} {'Input Tokens':<12} {'Output Tokens':<13}")
    print("-" * 80)
    
    for stats in all_stats:
        print(f"{stats['name']:<20} {stats['total_questions']:<10} {stats['correct_answers']:<8} "
              f"{stats['accuracy']:<10.1f} {stats['avg_latency']:<12.0f} "
              f"{stats['total_input_tokens']:<12,} {stats['total_output_tokens']:<13,}")
    
    print("="*80)

def main():
    """Main function to compile and visualize benchmark statistics"""
    print("Compiling benchmark statistics...")
    
    # Find all log files
    log_files = find_log_files()
    if not log_files:
        print("No valid log files found in the logs directory!")
        return
    
    print(f"Found {len(log_files)} log files:")
    for file in log_files:
        print(f"  - {file}")
    
    # Parse each log file
    all_stats = []
    for file_path in log_files:
        print(f"\nProcessing: {file_path}")
        stats = parse_jsonl_file(file_path)
        all_stats.append(stats)
        
        # Print basic info for each file
        print(f"  Name: {stats['name']}")
        print(f"  Questions: {stats['total_questions']}")
        print(f"  Accuracy: {stats['accuracy']:.1f}%")
        print(f"  Avg Latency: {stats['avg_latency']:.0f}ms")
        print(f"  Avg Input Tokens per Query: {stats['avg_input_tokens']:.1f}")
        print(f"  Avg Output Tokens per Query: {stats['avg_output_tokens']:.1f}")
        print(f"  Total Input Tokens: {stats['total_input_tokens']:,}")
        print(f"  Total Output Tokens: {stats['total_output_tokens']:,}")
    
    # Print summary table
    print_summary_table(all_stats)
    
    # Create comparison graphs
    print("\nGenerating comparison graphs...")
    create_comparison_graphs(all_stats)
    
    print("\nStatistics compilation complete!")

if __name__ == "__main__":
    main()