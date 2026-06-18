"""
Multi-Agent Research System - Main Entry Point

Command-line interface for running the research workflow.
Supports topic input and configuration options.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from config.settings import get_settings
from src.graph.workflow import run_research
from src.utils.logger import setup_logging, get_logger


def main():
    """Main entry point for the research system."""
    # Setup logging and settings
    settings = get_settings()
    
    parser = argparse.ArgumentParser(
        description="Multi-Agent Research System - Research technical topics and generate analysis reports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --topic "Vector Databases vs Traditional Databases"
  python main.py --topic "LangChain vs LlamaIndex" --max-iterations 5
  python main.py --topic "Kubernetes Best Practices" --verbose
        """
    )
    
    parser.add_argument(
        "--topic", "-t",
        type=str,
        required=True,
        help="Technical topic to research"
    )
    
    parser.add_argument(
        "--max-iterations", "-m",
        type=int,
        default=settings.max_iterations,
        help=f"Maximum research-analysis iterations (default: {settings.max_iterations})"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--json-logs",
        action="store_true",
        help="Output logs in JSON format"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else settings.log_level
    setup_logging(log_level=log_level, json_format=args.json_logs)
    
    logger = get_logger("main")
    
    logger.info(
        "Starting Multi-Agent Research System",
        topic=args.topic,
        max_iterations=args.max_iterations
    )
    
    try:
        # Run the research workflow
        result = run_research(
            topic=args.topic,
            max_iterations=args.max_iterations,
            verbose=args.verbose
        )
        
        # Print results
        print("\n" + "=" * 60)
        print("RESEARCH COMPLETE")
        print("=" * 60)
        
        status = result.get("status", "unknown")
        print(f"\nStatus: {status}")
        print(f"Iterations: {result.get('iterations', 0)}")
        print(f"Sources analyzed: {len(result.get('validated_sources', []))}")
        
        # Report paths
        report_paths = result.get("report_paths", {})
        if report_paths:
            print("\nGenerated Reports:")
            if report_paths.get("markdown"):
                print(f"  Markdown: {report_paths['markdown']}")
            if report_paths.get("pdf"):
                print(f"  PDF: {report_paths['pdf']}")
        
        # Errors
        errors = result.get("errors", [])
        if errors:
            print(f"\nWarnings/Errors ({len(errors)}):")
            for err in errors[:5]:  # Limit displayed errors
                print(f"  - {err}")
        
        # Print report preview
        final_report = result.get("final_report", "")
        if final_report:
            print("\n" + "-" * 60)
            print("REPORT PREVIEW (first 1000 chars):")
            print("-" * 60)
            print(final_report[:1000])
            if len(final_report) > 1000:
                print("\n... [truncated, see full report in output files]")
        
        print("\n" + "=" * 60)
        
        # Return appropriate exit code
        return 0 if status == "done" else 1
        
    except KeyboardInterrupt:
        logger.info("Research interrupted by user")
        return 130
        
    except Exception as e:
        logger.error("Research failed", error=str(e), error_type=type(e).__name__)
        print(f"\nERROR: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
