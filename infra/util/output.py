from domain.models.test_case import TestCase


def print_test_case(test_case: TestCase):
    title = '-' * 10 + 'CASO DE TESTE' + '-' * 10
    output = f"""
{title}
ID:             {test_case['id']}
Network:        {test_case['network']}
Backbone:       {test_case['backbone']}
Otimizador:     {test_case['optimizer']}
Pesos imagenet: {test_case['use_imagenet_weights']}
{'-' * len(title)}
    """
    print(output)
