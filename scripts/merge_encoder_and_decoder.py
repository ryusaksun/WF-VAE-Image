import argparse
import torch


def pick_model_state(ckpt: dict):
    ema_state = ckpt.get("ema_state_dict")
    if isinstance(ema_state, dict) and len(ema_state) > 0:
        return "ema_state_dict", ema_state

    state = ckpt.get("state_dict")
    if isinstance(state, dict):
        return "state_dict", state

    raise KeyError("Checkpoint must contain a non-empty `ema_state_dict` or a valid `state_dict`.")


def main():
    parser = argparse.ArgumentParser(description="Merge encoder and decoder checkpoints")
    parser.add_argument("--encoder", type=str, required=True, help="Path to encoder checkpoint")
    parser.add_argument("--decoder", type=str, required=True, help="Path to decoder checkpoint")
    parser.add_argument("--output", type=str, required=True, help="Path to save merged checkpoint")
    args = parser.parse_args()

    encoder_ckpt = torch.load(args.encoder, map_location="cpu")
    decoder_ckpt = torch.load(args.decoder, map_location="cpu")

    new_ckpt = encoder_ckpt

    _, decoder_state = pick_model_state(decoder_ckpt)
    target_key, target_state = pick_model_state(new_ckpt)

    for key, param in decoder_state.items():
        target_state[key] = param

    new_ckpt[target_key] = target_state

    torch.save(new_ckpt, args.output)


if __name__ == "__main__":
    main()
