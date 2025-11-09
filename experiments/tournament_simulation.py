"""
Tournament Simulation System for Go AI Models

Runs round-robin tournament between all models with comprehensive metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model_interface import (
    BaselineModel, PureMCTSModel, PolicyOnlyModel, RandomMCTSModel
)
from utils.go_game import GoGame


class TournamentSimulator:
    """Manages tournament between multiple Go AI models"""
    
    def __init__(self, models, games_per_matchup=50, save_dir="experiments/tournament_results"):
        self.models = models
        self.games_per_matchup = games_per_matchup
        self.save_dir = save_dir
        
        os.makedirs(save_dir, exist_ok=True)
        
        self.results = {
            'games': [],
            'win_matrix': None,
            'stats': {}
        }
        
    def run_tournament(self):
        print("=" * 70)
        print("GO AI TOURNAMENT SIMULATION")
        print("=" * 70)
        print(f"\nModels: {[m.name for m in self.models]}")
        print(f"Games per matchup: {self.games_per_matchup}")
        total_games = len(self.models) * (len(self.models) - 1) // 2 * self.games_per_matchup
        print(f"Total games: {total_games}")
        print("=" * 70)
        
        matchup_count = 0
        total_matchups = len(self.models) * (len(self.models) - 1) // 2
        
        for i in range(len(self.models)):
            for j in range(i + 1, len(self.models)):
                matchup_count += 1
                model1 = self.models[i]
                model2 = self.models[j]
                
                print(f"\n{'='*70}")
                print(f"MATCHUP {matchup_count}/{total_matchups}: {model1.name} vs {model2.name}")
                print(f"{'='*70}")
                
                matchup_results = self.run_matchup(model1, model2)
                
                wins_m1 = sum(1 for r in matchup_results if r['winner'] == model1.name)
                wins_m2 = sum(1 for r in matchup_results if r['winner'] == model2.name)
                draws = sum(1 for r in matchup_results if r['winner'] == 'Draw')
                
                print(f"\nMatchup Results:")
                print(f"  {model1.name}: {wins_m1} wins ({wins_m1/self.games_per_matchup*100:.1f}%)")
                print(f"  {model2.name}: {wins_m2} wins ({wins_m2/self.games_per_matchup*100:.1f}%)")
                print(f"  Draws: {draws}")
        
        self._compute_statistics()
        self._save_results()
        self._generate_visualizations()
        
        print("\n" + "=" * 70)
        print("TOURNAMENT COMPLETED!")
        print("=" * 70)
        
        return self.results
    
    def run_matchup(self, model1, model2):
        matchup_results = []
        
        for game_num in range(self.games_per_matchup):
            if game_num % 2 == 0:
                black_model, white_model = model1, model2
            else:
                black_model, white_model = model2, model1
            
            print(f"\nGame {game_num + 1}/{self.games_per_matchup}: ", end="")
            print(f"Black={black_model.name}, White={white_model.name}")
            
            game_result = self.play_game(black_model, white_model)
            matchup_results.append(game_result)
            self.results['games'].append(game_result)
            
            winner = game_result['winner']
            moves = game_result['num_moves']
            duration = game_result['game_duration']
            print(f"  Winner: {winner}, Moves: {moves}, Duration: {duration:.2f}s")
        
        return matchup_results
    
    def play_game(self, black_model, white_model, max_moves=200):
        game = GoGame()
        game_data = {
            'black_model': black_model.name,
            'white_model': white_model.name,
            'moves': [],
            'move_times': {'black': [], 'white': []},
            'timestamp': datetime.now().isoformat()
        }
        
        start_time = time.time()
        move_count = 0
        
        while move_count < max_moves:
            current_model = black_model if game.current_player == GoGame.BLACK else white_model
            player_name = 'black' if game.current_player == GoGame.BLACK else 'white'
            
            move_start = time.time()
            move = current_model.select_move(game)
            move_time = time.time() - move_start
            
            game_data['move_times'][player_name].append(move_time)
            game_data['moves'].append({
                'move_num': move_count,
                'player': game.current_player,
                'move': move,
                'time': move_time
            })
            
            done, reward = game.make_move(move)
            move_count += 1
            
            if done:
                score = game._calculate_score()
                
                if score > 0:
                    winner = black_model.name
                elif score < 0:
                    winner = white_model.name
                else:
                    winner = 'Draw'
                
                game_data.update({
                    'winner': winner,
                    'final_score': score,
                    'num_moves': move_count,
                    'game_duration': time.time() - start_time,
                    'avg_move_time_black': np.mean(game_data['move_times']['black']),
                    'avg_move_time_white': np.mean(game_data['move_times']['white']),
                    'completed': True
                })
                
                return game_data
        
        score = game._calculate_score()
        if score > 0:
            winner = black_model.name
        elif score < 0:
            winner = white_model.name
        else:
            winner = 'Draw'
        
        game_data.update({
            'winner': winner,
            'final_score': score,
            'num_moves': move_count,
            'game_duration': time.time() - start_time,
            'avg_move_time_black': np.mean(game_data['move_times']['black']),
            'avg_move_time_white': np.mean(game_data['move_times']['white']),
            'completed': False,
            'reason': 'max_moves'
        })
        
        return game_data
    
    def _compute_statistics(self):
        model_names = [m.name for m in self.models]
        n_models = len(model_names)
        
        win_matrix = np.zeros((n_models, n_models))
        
        for game in self.results['games']:
            black = game['black_model']
            white = game['white_model']
            winner = game['winner']
            
            black_idx = model_names.index(black)
            white_idx = model_names.index(white)
            
            if winner == black:
                win_matrix[black_idx, white_idx] += 1
            elif winner == white:
                win_matrix[white_idx, black_idx] += 1
        
        self.results['win_matrix'] = win_matrix
        
        for model_name in model_names:
            model_games = [g for g in self.results['games'] 
                          if g['black_model'] == model_name or g['white_model'] == model_name]
            
            wins = sum(1 for g in model_games if g['winner'] == model_name)
            total = len(model_games)
            
            black_times = [g['avg_move_time_black'] for g in model_games 
                          if g['black_model'] == model_name]
            white_times = [g['avg_move_time_white'] for g in model_games 
                          if g['white_model'] == model_name]
            all_times = black_times + white_times
            
            game_lengths = [g['num_moves'] for g in model_games]
            
            self.results['stats'][model_name] = {
                'total_games': total,
                'wins': wins,
                'losses': total - wins,
                'win_rate': wins / total if total > 0 else 0,
                'avg_game_length': np.mean(game_lengths),
                'avg_move_time': np.mean(all_times),
                'std_move_time': np.std(all_times),
                'total_time': sum(all_times)
            }
        
        self._compute_elo_ratings()
    
    def _compute_elo_ratings(self, k=32, base_rating=1500):
        model_names = [m.name for m in self.models]
        ratings = {name: base_rating for name in model_names}
        
        def expected_score(rating_a, rating_b):
            return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
        
        for game in self.results['games']:
            black = game['black_model']
            white = game['white_model']
            winner = game['winner']
            
            if winner == black:
                score_black, score_white = 1, 0
            elif winner == white:
                score_black, score_white = 0, 1
            else:
                score_black, score_white = 0.5, 0.5
            
            exp_black = expected_score(ratings[black], ratings[white])
            exp_white = expected_score(ratings[white], ratings[black])
            
            ratings[black] += k * (score_black - exp_black)
            ratings[white] += k * (score_white - exp_white)
        
        for model_name in model_names:
            self.results['stats'][model_name]['elo_rating'] = round(ratings[model_name], 1)
    
    def _save_results(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        games_file = os.path.join(self.save_dir, f"games_{timestamp}.json")
        with open(games_file, 'w') as f:
            games_data = []
            for game in self.results['games']:
                game_copy = game.copy()
                games_data.append(game_copy)
            json.dump(games_data, f, indent=2)
        
        stats_df = pd.DataFrame(self.results['stats']).T
        stats_file = os.path.join(self.save_dir, f"statistics_{timestamp}.csv")
        stats_df.to_csv(stats_file)
        
        model_names = [m.name for m in self.models]
        win_df = pd.DataFrame(
            self.results['win_matrix'],
            index=model_names,
            columns=model_names
        )
        matrix_file = os.path.join(self.save_dir, f"win_matrix_{timestamp}.csv")
        win_df.to_csv(matrix_file)
        
        print(f"\n✓ Results saved to {self.save_dir}")
        print(f"  - Games: {games_file}")
        print(f"  - Statistics: {stats_file}")
        print(f"  - Win Matrix: {matrix_file}")
    
    def _generate_visualizations(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_names = [m.name for m in self.models]
        
        fig = plt.figure(figsize=(16, 10))
        
        # Win rates
        ax1 = plt.subplot(2, 3, 1)
        win_rates = [self.results['stats'][name]['win_rate'] * 100 
                     for name in model_names]
        ax1.bar(model_names, win_rates, color='steelblue')
        ax1.set_ylabel('Win Rate (%)')
        ax1.set_title('Overall Win Rates')
        ax1.set_ylim(0, 100)
        for i, v in enumerate(win_rates):
            ax1.text(i, v + 2, f'{v:.1f}%', ha='center')
        
        # Win matrix
        ax2 = plt.subplot(2, 3, 2)
        sns.heatmap(self.results['win_matrix'], 
                   annot=True, fmt='.0f', 
                   xticklabels=model_names,
                   yticklabels=model_names,
                   cmap='YlOrRd', ax=ax2)
        ax2.set_title('Head-to-Head Wins')
        
        # ELO ratings
        ax3 = plt.subplot(2, 3, 3)
        elo_ratings = [self.results['stats'][name]['elo_rating'] 
                      for name in model_names]
        ax3.barh(model_names, elo_ratings, color=['gold', 'silver', 'brown', 'gray'][:len(model_names)])
        ax3.set_xlabel('ELO Rating')
        ax3.set_title('ELO Ratings')
        
        # Move times
        ax4 = plt.subplot(2, 3, 4)
        move_times = [self.results['stats'][name]['avg_move_time'] 
                     for name in model_names]
        ax4.bar(model_names, move_times, color='coral')
        ax4.set_ylabel('Average Time (seconds)')
        ax4.set_title('Average Move Time')
        
        # Game lengths
        ax5 = plt.subplot(2, 3, 5)
        game_lengths = [self.results['stats'][name]['avg_game_length'] 
                       for name in model_names]
        ax5.bar(model_names, game_lengths, color='lightgreen')
        ax5.set_ylabel('Moves')
        ax5.set_title('Average Game Length')
        
        # Wins vs losses
        ax6 = plt.subplot(2, 3, 6)
        wins = [self.results['stats'][name]['wins'] for name in model_names]
        losses = [self.results['stats'][name]['losses'] for name in model_names]
        x = np.arange(len(model_names))
        width = 0.35
        ax6.bar(x - width/2, wins, width, label='Wins', color='green')
        ax6.bar(x + width/2, losses, width, label='Losses', color='red')
        ax6.set_ylabel('Count')
        ax6.set_title('Wins vs Losses')
        ax6.set_xticks(x)
        ax6.set_xticklabels(model_names)
        ax6.legend()
        
        plt.tight_layout()
        
        viz_file = os.path.join(self.save_dir, f"tournament_results_{timestamp}.png")
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        print(f"  - Visualizations: {viz_file}")
        
        plt.close()
    
    def print_final_summary(self):
        print("\n" + "=" * 70)
        print("FINAL TOURNAMENT SUMMARY")
        print("=" * 70)
        
        model_names = sorted(
            [m.name for m in self.models],
            key=lambda name: self.results['stats'][name]['elo_rating'],
            reverse=True
        )
        
        print("\nFinal Rankings (by ELO):")
        for i, name in enumerate(model_names, 1):
            stats = self.results['stats'][name]
            print(f"\n{i}. {name}")
            print(f"   ELO Rating: {stats['elo_rating']:.1f}")
            print(f"   Win Rate: {stats['win_rate']*100:.1f}%")
            print(f"   Record: {stats['wins']}-{stats['losses']}")
            print(f"   Avg Move Time: {stats['avg_move_time']:.3f}s")
        
        print("\n" + "=" * 70)


def main():
    print("Initializing tournament...")
    
    models = [
        BaselineModel(num_simulations=100, device='cpu'),
        PureMCTSModel(num_simulations=100),
        PolicyOnlyModel(device='cpu'),
        RandomMCTSModel(num_simulations=100)
    ]
    
    tournament = TournamentSimulator(
        models=models,
        games_per_matchup=50,
        save_dir="experiments/tournament_results"
    )
    
    results = tournament.run_tournament()
    tournament.print_final_summary()
    
    print("\n✓ Tournament complete!")


if __name__ == "__main__":
    main()
