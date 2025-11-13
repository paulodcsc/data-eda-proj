# Definição do Problema

## Contexto da Olist
A Olist oferece uma plataforma de comércio eletrônico que conecta lojistas brasileiros a marketplaces de grande alcance. Como operação logística, a empresa precisa garantir entregas rápidas e confiáveis para manter a satisfação dos consumidores, reduzindo atrasos e custos. Prever quanto tempo um pedido leva para chegar ao cliente permite priorizar rotas, ajustar prazos estimados e sinalizar pedidos em risco.

## Problema de Regressão
Cada amostra representa um pedido completo: compra, itens, seller(es), cliente, pagamento, revisões e localização. A variável alvo é o tempo real de entrega, medido em dias úteis (diferença entre a data da compra e a entrega ao cliente). Nosso modelo de regressão tentará estimar essa diferença com base nas características disponíveis antes da entrega final.

## Estrutura dos Dados
Trabalhamos com as tabelas oficiais da Olist:
- `orders`: linhas em formato de pedido, com timestamps importantes (compra, aprovação, entrega estimada e real);
- `order_items`: nível de item, preços, frete e relacionamentos com sellers e produtos;
- `customers` e `sellers`: informações de cliente/seller incluindo CEP e estado;
- `geolocation`: latitudes e longitudes associadas a CEPs, usadas para contextualizar a distância entre seller e cliente;
- `payments` e `reviews`: formas de pagamento, parcelas e avaliação do pedido.

## Desafios do Problema
Prever tempo de entrega reforça um desafio real da logística: as rotas envolvem múltiplos sellers, condições geográficas distintas e variações de pagamento/quantidade de itens. A interdependência entre timestamps também exige cuidado no tratamento de dados faltantes e derivação de features. Modelar esse cenário ajuda a evitar promessas de prazo incorretas, melhora o planejamento e reduz custos de suporte quando o pedido atrasa.
