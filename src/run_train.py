import argpasrse

from model import RNNTextModel
from text import Progress, read_data_files


def _main():
    args = _parse_args()

    training_text, validation_text, file_index = read_data_files(
        args.files, validation=True)

    sequence_length = 30
    batch_size = 200
    display_every_n_batches = 50
    progress = Progress(
        display_every_n_batches,
        size=111+2,
        msg=f'Training on next {display_every_n_batches} batches')

    def on_step_completed(step: int):
        # TODO: explain
        max_steps = display_every_n_batches * batch_size * sequence_length
        progress.step(reset=step % max_steps == 0)

    model = RNNTextModel(
        sequence_length=sequence_length,
        gru_internal_size=512,
        num_hidden_layers=3,
        stats_log_dir='log')
    model.run_training_loop(
        training_text,
        num_epochs=10000,
        learning_rate=0.001,
        display_every_n_batches=50,
        batch_size=batch_size,
        checkpoint_dir='checkpoints',
        on_step_complete=on_step_complete,
        should_stop=lambda step: False)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-f', '--files', type=str,
        description='TODO')
    return parser.parse_args()


if __name__ == '__main__':
    _main()
