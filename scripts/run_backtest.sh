#!/bin/bash
# Script helper pour lancer des backtests facilement

# Couleurs
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Backtest Runner ===${NC}\n"

# Vérifier si un argument est fourni
if [ "$1" == "--list" ] || [ "$1" == "-l" ]; then
    echo -e "${GREEN}Liste des stratégies disponibles:${NC}"
    python scripts/run_backtest.py --list-strategies
    exit 0
fi

if [ -z "$1" ]; then
    echo -e "${RED}Usage:${NC}"
    echo "  $0 <config_file>           # Lance un backtest avec le fichier config"
    echo "  $0 --list                  # Liste les stratégies disponibles"
    echo ""
    echo -e "${BLUE}Exemples:${NC}"
    echo "  $0 config/backtest_config.yaml"
    echo "  $0 config/examples/example_rsi_config.yaml"
    exit 1
fi

CONFIG_FILE="$1"

if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}Erreur: Fichier de configuration introuvable: $CONFIG_FILE${NC}"
    exit 1
fi

echo -e "${GREEN}Lancement du backtest avec: $CONFIG_FILE${NC}\n"
python scripts/run_backtest.py --config "$CONFIG_FILE"
