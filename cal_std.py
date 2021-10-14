import click
import glob


@click.command()
@click.option('--prefix', '-p')
@click.option('--number', '-m', multiple=True)
def start(prefix, number):
    print(prefix, number)
    filenames = map(lambda n: '{}-{}.out'.format(prefix, n), number)
    parameters = set()
    val_score = []
    test_score = []
    for file_name in filenames:
        with open(file_name) as file:
            for line in file:
                if 'Namespace' in line:
                    parameters.add(line)
                elif 'Best validation score' in line:
                    val_score.append(float(line.split(': ')[-1]))
                elif 'Test score' in line:
                    test_score.append(float(line.split(': ')[-1]))
    print('Parameters:')
    for p in parameters:
        print(p)

    import torch
    val_score = torch.tensor(val_score)
    test_score = torch.tensor(test_score)

    print('Val score: {} +- {}'.format(val_score.mean().item(), val_score.std().item()))
    print('Test score: {} +- {}'.format(test_score.mean().item(), test_score.std().item()))


if __name__ == '__main__':
    start()
