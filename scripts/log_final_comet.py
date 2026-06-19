import getpass
import os

from comet_ml import Experiment


RUN = "modular_prepost5_final_full_test"
PROJECT = "hw05"

PSNR = [
    12.246738197281957,
    12.844870837405324,
    13.171467205509543,
    13.355594046413898,
    13.512064971029758,
    13.627942252904177,
    13.712385542690754,
    13.771761901676655,
    13.828378677368164,
    13.870173271745443,
    13.905605897307396,
    13.939018294215202,
    13.967441320419312,
    13.99821500852704,
    14.027169838547707,
    14.056157629936934,
    14.081013202667236,
    14.116707991808653,
    14.126502465456724,
    14.152297474443913,
    14.17269615828991,
    14.196448456496,
    14.216738555580378,
    14.242087315768003,
    14.253501038998365,
    14.273214306682348,
    14.289286997169256,
    14.306988220661879,
    14.318113092333078,
]


def main():
    key = os.environ.get("COMET_API_KEY") or getpass.getpass("COMET_API_KEY: ")
    exp = Experiment(api_key=key, project_name=PROJECT)
    exp.set_name(RUN)
    exp.add_tag("final")
    exp.add_tag("modular_leadmm")
    exp.log_parameters(
        {
            "model": "modular",
            "steps": 5,
            "base": 64,
            "size": 256,
            "batch": 4,
            "lr": 2e-4,
            "l1": 0.1,
            "ema": 0.999,
            "quick_val_limit": 256,
            "full_test_count": 1500,
        }
    )
    best = 0.0
    best_epoch = 0
    for epoch, value in enumerate(PSNR):
        best = max(best, value)
        if best == value:
            best_epoch = epoch
        exp.log_metric("psnr_quick_val256", value, step=epoch, epoch=epoch)
        exp.log_metric("best_psnr_quick_val256", best, step=epoch, epoch=epoch)
    exp.log_metric("full_test_psnr", 14.675472382863363, step=len(PSNR), epoch=best_epoch)
    exp.log_metric("best_psnr", 14.675472382863363, step=len(PSNR), epoch=best_epoch)
    exp.log_metric("best_epoch", best_epoch)
    exp.log_other("checkpoint_url", "https://github.com/boldirev-as/hw05/releases/download/checkpoint-v1/best.pt")
    exp.log_other("demo_data_url", "https://github.com/boldirev-as/hw05/releases/download/checkpoint-v1/demo_data.zip")
    exp.log_other("note", "Training curve reconstructed from the final Modular LeADMM cluster log; full_test_psnr was measured on the complete test split.")
    print(exp.url)
    exp.end()


if __name__ == "__main__":
    main()
