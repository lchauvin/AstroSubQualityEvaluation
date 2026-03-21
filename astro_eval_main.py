"""PyInstaller entry point — runs astro_eval.cli:main as a proper package."""
import multiprocessing

if __name__ == "__main__":
    multiprocessing.freeze_support()
    from astro_eval.cli import main
    main()
