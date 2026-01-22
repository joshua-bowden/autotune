"""
Service orchestration module for the radio archiver system.

This module starts and manages all radio archiver services:
- Unified Streaming Engine (Capture, Archiver, Transcription)
- Story processing and embedding using Gemini

All services run as separate processes and are monitored for health.
The orchestrator handles graceful shutdown and basic process recovery.
"""

import logging
import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple

import config
import database
from utils import setup_logging


logger = setup_logging(__name__)


# Service monitoring constants
HEALTH_CHECK_INTERVAL = 5  # Seconds between health checks
RESTART_DELAY = 10  # Seconds to wait before restarting failed service
MAX_RESTART_ATTEMPTS = 3  # Maximum restart attempts per service


class ServiceManager:
    """Manages multiple service processes with health monitoring."""
    
    def __init__(self):
        """Initialize the service manager."""
        self.processes: List[Tuple[subprocess.Popen, str, int]] = []
        self.base_dir = Path(__file__).parent
    
    def start_service(self, script_name: str, service_name: str) -> None:
        """
        Start a service process.
        
        Args:
            script_name: Name of the Python script to run
            service_name: Human-readable service name for logging
        """
        script_path = self.base_dir / script_name
        
        if not script_path.exists():
            logger.error(f"Service script not found: {script_path}")
            return
        
        try:
            logger.info(f"Starting {service_name}...")
            # Allow real-time output by not capturing stdout/stderr
            process = subprocess.Popen(
                [sys.executable, str(script_path)],
                stdout=None,  # Inherit parent's stdout
                stderr=None   # Inherit parent's stderr
            )
            self.processes.append((process, service_name, 0))  # (process, name, restart_count)
            logger.info(f"{service_name} started (PID: {process.pid})")
        except Exception as e:
            logger.error(f"Failed to start {service_name}: {e}")
    
    def check_health(self) -> None:
        """
        Check health of all services and restart if needed.
        
        Monitors process status and restarts failed services up to
        MAX_RESTART_ATTEMPTS times.
        """
        for i, (process, name, restart_count) in enumerate(self.processes):
            if process.poll() is not None:
                # Process has exited
                exit_code = process.returncode
                logger.warning(f"{name} exited with code {exit_code}")
                
                if restart_count < MAX_RESTART_ATTEMPTS:
                    logger.info(
                        f"Restarting {name} (attempt {restart_count + 1}/{MAX_RESTART_ATTEMPTS})..."
                    )
                    time.sleep(RESTART_DELAY)
                    
                    # Determine script name from service name
                    script_map = {
                        "Streaming Engine": "engine.py",
                        "Live Transcriber": "live_transcriber.py",
                        "Story Processing": "processor.py"
                    }
                    script_name = script_map.get(name)
                    
                    if script_name:
                        script_path = self.base_dir / script_name
                        try:
                            new_process = subprocess.Popen(
                                [sys.executable, str(script_path)],
                                stdout=None,  # Inherit parent's stdout
                                stderr=None   # Inherit parent's stderr
                            )
                            self.processes[i] = (new_process, name, restart_count + 1)
                            logger.info(f"{name} restarted (PID: {new_process.pid})")
                        except Exception as e:
                            logger.error(f"Failed to restart {name}: {e}")
                else:
                    logger.error(
                        f"{name} has failed {MAX_RESTART_ATTEMPTS} times, "
                        "not restarting"
                    )
    
    def stop_all(self) -> None:
        """Stop all running services gracefully."""
        logger.info("Stopping all services...")
        
        for process, name, _ in self.processes:
            if process.poll() is None:
                logger.info(f"Stopping {name}...")
                process.terminate()
                
                # Wait up to 5 seconds for graceful shutdown
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning(f"{name} did not stop gracefully, forcing...")
                    process.kill()
        
        logger.info("All services stopped")


def run_services(live_mode: bool = False) -> None:
    """
    Start and monitor all radio archiver services.
    
    Runs until interrupted by user (Ctrl+C) or critical error.
    Automatically restarts failed services up to MAX_RESTART_ATTEMPTS times.
    """
    logger.info("=" * 70)
    logger.info("Radio Archiver Service Orchestrator")
    logger.info("=" * 70)
    logger.info("")
    logger.info("WHAT GETS EMBEDDED:")
    logger.info("  - Each story's summary + full transcript text (concatenated)")
    logger.info("  - Gemini segments 5-minute batches into distinct stories")
    logger.info("  - Each story gets its own 768-dimensional embedding vector")
    logger.info("")
    logger.info("WHAT GETS STORED IN DATABASE:")
    logger.info("  - Story transcript (full text)")
    logger.info("  - Story summary (AI-generated title)")
    logger.info("  - Audio metadata (filenames + timestamps)")
    logger.info("  - Embedding vector (for semantic search)")
    logger.info("")
    logger.info("SERVICES:")
    logger.info("  1. Streaming Engine  - Real-time capture, archiving, and transcription")
    logger.info("  2. Story Processing  - Segments, summarizes, embeds using Gemini")
    logger.info("=" * 70)
    logger.info("")
    
    manager = ServiceManager()
    
    # Ensure database is initialized
    logger.info("Initializing database...")
    if not database.init_db():
        logger.error("Failed to initialize database. Exiting.")
        return

    # Start all services
    if live_mode:
        manager.start_service("live_transcriber.py", "Live Transcriber")
    else:
        manager.start_service("engine.py", "Streaming Engine")
    
    manager.start_service("processor.py", "Story Processing")
    
    logger.info("\nAll services started successfully")
    logger.info("Press Ctrl+C to stop all services\n")
    
    try:
        while True:
            time.sleep(HEALTH_CHECK_INTERVAL)
            manager.check_health()
            
    except KeyboardInterrupt:
        logger.info("\nShutdown requested by user")
    except Exception as e:
        logger.error(f"Unexpected error in orchestrator: {e}")
    finally:
        manager.stop_all()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Radio Archiver Service Orchestrator")
    parser.add_argument("--live", action="store_true", help="Use live microphone transcription instead of stream")
    args = parser.parse_args()
    
    run_services(live_mode=args.live)
