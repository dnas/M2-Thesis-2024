// Copyright 2019 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "open_spiel/games/postflop/postflop.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <numeric>
#include <utility>
#include <iostream>
#include <fstream>

#include "open_spiel/abseil-cpp/absl/algorithm/container.h"
#include "open_spiel/abseil-cpp/absl/memory/memory.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/observer.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

#define EPS 0.01
#define PLACEHOLDER -100

namespace open_spiel {
namespace postflop {
namespace {

//int counter_zzz = 0;
std::vector<int> kBlinds = {5, 10}; //Button and Big Blind. The button acts first pre-flop, the BB acts first post-flop

const GameType kGameType{/*short_name=*/"postflop",
                         /*long_name=*/"Postflop Pot Limit Omaha",
                         GameType::Dynamics::kSequential,
                         GameType::ChanceMode::kExplicitStochastic,
                         GameType::Information::kImperfectInformation,
                         GameType::Utility::kZeroSum,
                         GameType::RewardModel::kTerminal,
                         /*max_num_players=*/2,
                         /*min_num_players=*/2,
                         /*provides_information_state_string=*/true,
                         /*provides_information_state_tensor=*/true,
                         /*provides_observation_string=*/true,
                         /*provides_observation_tensor=*/true,
                         /*parameter_specification=*/
                         {{"players", GameParameter(kDefaultPlayers)},
                         {"num_buckets", GameParameter(kDefaultBuckets)}}};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new PostflopGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

std::string StatelessActionToString(Action action) {
  if (action == ActionType::kF) {
    return "Fold";
  } else if (action == ActionType::kX) {
    return "Check";
  } else if (action == ActionType::kC) {
    return "Call";
  } else if (action == ActionType::kB) {
    return "Bet";
  } else if (action == ActionType::kR) {
    return "Raise";
  } else {
    SpielFatalError(absl::StrCat("Unknown action: ", action));
  }
  return "Will not return.";
}

RegisterSingleTensorObserver single_tensor(kGameType.short_name);
}  // namespace

// The Observer class is responsible for creating representations of the game
// state for use in learning algorithms. It handles both string and tensor
// representations, and any combination of public information and private
// information (none, observing player only, or all players).
//
// If a perfect recall observation is requested, it must be possible to deduce
// all previous observations for the same information type from the current
// observation.

class PostflopObserver : public Observer {
 public:
  PostflopObserver(IIGObservationType iig_obs_type)
      : Observer(/*has_string=*/true, /*has_tensor=*/true),
        iig_obs_type_(iig_obs_type) {}

  //
  // These helper methods each write a piece of the tensor observation.
  //

  // Identity of the observing player. One-hot vector of size num_players.
  static void WriteObservingPlayer(const PostflopState& state, int player,
                                   Allocator* allocator) {
    auto out = allocator->Get("player", {state.num_players_});
    out.at(player) = 1;
  }

  //TO FIX
  // Private card of the observing player. One-hot vector of size num_cards.
  static void WriteSinglePlayerBucket(const PostflopState& state, int player,
                                    Allocator* allocator) {
    auto out = allocator->Get("private_bucket", {10});
    int bucket;
    if(state.round_==0) bucket = state.private_bucket_pre_[player];
    else bucket = state.private_bucket_post_[player];
    if (bucket != -1) out.at(bucket) = 1;
  }

  // Private cards of all players. Tensor of shape [num_players, num_cards].
  static void WriteAllPlayerBuckets(const PostflopState& state,
                                  Allocator* allocator) {
    auto out = allocator->Get("private_bucket",
                              {state.num_players_, 10});
    for (int p = 0; p < state.num_players_; ++p) {
      int bucket;
      if(state.round_==0){
        bucket = state.private_bucket_pre_[p];
        if (bucket != -1) out.at(p, state.private_bucket_pre_[p]) = 1;
      }else{
        bucket = state.private_bucket_post_[p];
        if (bucket != -1) out.at(p, state.private_bucket_post_[p]) = 1;
      }
    }
  }

  /*
  // Community cards. One-hot vector of size num_cards.
  static void WriteCommunityCards(const PostflopState& state,
                                 Allocator* allocator) {
    auto out = allocator->Get("community_card", {270725});
    if (state.public_cards_ != std::vector<Card>{kInvalidCard, kInvalidCard, kInvalidCard}) {
      out.at(state.public_cards_[0].rank, state.public_cards_[1].rank, state.public_cards_[2].rank) = 1;
    }
  }
  */

  // TO FIX
  // Betting sequence; shape [num_rounds, bets_per_round, num_actions].
  static void WriteBettingSequence(const PostflopState& state,
                                   Allocator* allocator) {
    const int kNumRounds = 2;
    const int kBitsPerAction = 10;
    const int max_bets_per_round = state.max_raises_;
    auto out = allocator->Get("betting",
                              {kNumRounds, max_bets_per_round, kBitsPerAction});
    for (int round : {0, 1}) {
      const auto& bets =
          (round == 0) ? state.round0_sequence_ : state.round1_sequence_;
      for (int i = 0; i < bets.size(); ++i) {
        if (bets[i] == ActionType::kC) {
          out.at(round, i, 0) = 1;  // Encode call as 10.
        } else if (bets[i] == ActionType::kR) {
          out.at(round, i, 1) = 1;  // Encode raise as 01.
        }
      }
    }
  }

  // Pot contribution per player (integer per player).
  static void WritePotContribution(const PostflopState& state,
                                   Allocator* allocator) {
    auto out = allocator->Get("pot_contribution", {state.num_players_});
    for (auto p = Player{0}; p < state.num_players_; p++) {
      out.at(p) = state.ante_[p];
    }
  }

  // Writes the complete observation in tensor form.
  // The supplied allocator is responsible for providing memory to write the
  // observation into.
  void WriteTensor(const State& observed_state, int player,
                   Allocator* allocator) const override {
    auto& state = open_spiel::down_cast<const PostflopState&>(observed_state);
    SPIEL_CHECK_GE(player, 0);
    SPIEL_CHECK_LT(player, state.num_players_);

    // Observing player.
    WriteObservingPlayer(state, player, allocator);

    // Private card(s).
    if (iig_obs_type_.private_info == PrivateInfoType::kSinglePlayer) {
      WriteSinglePlayerBucket(state, player, allocator);
    } else if (iig_obs_type_.private_info == PrivateInfoType::kAllPlayers) {
      WriteAllPlayerBuckets(state, allocator);
    }

    // Public information.
    if (iig_obs_type_.public_info) {
      iig_obs_type_.perfect_recall ? WriteBettingSequence(state, allocator)
                                   : WritePotContribution(state, allocator);
    }
  }

  // Writes an observation in string form. It would be possible just to
  // turn the tensor observation into a string, but we prefer something
  // somewhat human-readable.

  std::string StringFrom(const State& observed_state,
                         int player) const override {
    auto& state = open_spiel::down_cast<const PostflopState&>(observed_state);
    SPIEL_CHECK_GE(player, 0);
    SPIEL_CHECK_LT(player, state.num_players_);
    std::string result;

    // Private card(s).
    if (iig_obs_type_.private_info == PrivateInfoType::kSinglePlayer) {
      absl::StrAppend(&result, "[Observer: ", player, "]");
      absl::StrAppend(&result, "[Private pre: ", state.private_bucket_pre_[player], "]");
      absl::StrAppend(&result, "[Private post: ", state.private_bucket_post_[player], "]");
    } else if (iig_obs_type_.private_info == PrivateInfoType::kAllPlayers) {
      absl::StrAppend(&result, "[Privates pre: ", state.private_bucket_pre_[0], " ", state.private_bucket_pre_[1], "]");
      absl::StrAppend(&result, "[Privates post: ", state.private_bucket_post_[0], " ", state.private_bucket_post_[1], "]");
    }

    // Public info. Not all of this is strictly necessary, but it makes the
    // string easier to understand.
    if (iig_obs_type_.public_info) {
      absl::StrAppend(&result, "[Round ", state.round_, "]");
      absl::StrAppend(&result, "[Player: ", state.cur_player_, "]");
      absl::StrAppend(&result, "[Pot: ", state.pot_, "]");
      absl::StrAppend(&result, "[Stack: ", absl::StrJoin(state.stack_, " "), "]");
      if (iig_obs_type_.perfect_recall) {
        // Betting Sequence (for the perfect recall case)
        absl::StrAppend(&result, "[Round0: ", absl::StrJoin(state.round0_sequence_, " "),"][Round1: ", absl::StrJoin(state.round1_sequence_, " "), "]");
      } else {
        // Pot contributions (imperfect recall)
        absl::StrAppend(&result, "[Ante: ", absl::StrJoin(state.ante_, " "),
                        "]");
      }
    }

    // Done.
    return result;
  }

 private:
  IIGObservationType iig_obs_type_;
};

PostflopState::PostflopState(std::shared_ptr<const Game> game, int num_buckets)
    : State(game),
      cur_player_(kChancePlayerId),
      round_(0),   // Round number (0 or 1 - postflop or postflop).
      num_winners_(-1),
      pot_(kBlinds[0]+kBlinds[1]),  // Number of chips in the pot.
      action_is_closed_(false),
      last_to_act_(1),
      cur_max_bet_(kBlinds[1]),
      private_bucket_pre_dealt_(0),
      private_bucket_post_dealt_(0),
      players_remaining_(game->NumPlayers()),
      // Is this player a winner? Indexed by pid.
      winner_(game->NumPlayers(), false),
      // Each player's single bucket. Indexed by pid.
      private_bucket_pre_(game->NumPlayers(), -1),
      private_bucket_post_(game->NumPlayers(), -1),
      num_buckets_(num_buckets),
      // How much money each player has, indexed by pid.
      stack_(game->NumPlayers()),
      // How much each player has contributed to the pot, indexed by pid.
      ante_(game->NumPlayers()),
      cur_round_bet_(game->NumPlayers(), 0.0),
      // Sequence of actions for each round. Needed to report information
      // state.
      round0_sequence_(),
      round1_sequence_(){
      // Players cannot distinguish between cards of different suits with the
      // same rank.{
  // Cards by value (0-6 for standard 2-player game, kInvalidCard if no longer
  // in the deck.)
  std::iota(players_remaining_.begin(), players_remaining_.end(), 0);
	for(int p = 0;p<game->NumPlayers();p++){
		stack_[p] = kDefaultStacks-kBlinds[p];
		ante_[p] = kBlinds[p];
    cur_round_bet_[p] = kBlinds[p];
	}
  nr_raises_ = 0;
  max_raises_ = 3; //should be infinity
  postflop_prob_.resize(num_buckets_);
  std::ifstream ifile;
  ifile.open("postflop_buckets_prob.txt");
  for(int i=0;i<num_buckets_;i++) ifile >> postflop_prob_[i];
  ifile.close();
  transition_matrix_.resize(num_buckets_);
  ifile.open("postflop_buckets_matrix.txt");
  for(int i=0;i<num_buckets_;i++){
    transition_matrix_[i].resize(num_buckets_);
    for(int j=0;j<num_buckets_;j++) ifile >> transition_matrix_[i][j];
  }
  ifile.close();
}

int PostflopState::CurrentPlayer() const {
  if (IsTerminal()) {
    return kTerminalPlayerId;
  } else {
    return cur_player_;
  }
}

// In a chance node, `move` should be the card to deal to the current
// underlying player.
// On a player node, it should be ActionType::{kF, kX, kB, kC, kR}
void PostflopState::DoApplyAction(Action move) {
  //std::cout << "-----------------------------------------------------------------" << std::endl;
  //std::cout << "At DoApplyAction, round_ = " << round_ << ", cur_player_ = " << cur_player_ << ", Chance? " << IsChanceNode() << ", move = " << move << std::endl;
  if (IsChanceNode()) {
    SPIEL_CHECK_GE(move, 0);
    SPIEL_CHECK_LT(move, num_buckets_);

    if (round_==0&&private_bucket_pre_dealt_ < num_players_) { //round 0 - postflop
      SetPrivate(private_bucket_pre_dealt_, move);
      // When all private cards are dealt, move to player 0 (the Button/Small Blind, who acts first postflop).
      if (private_bucket_pre_dealt_ == num_players_) cur_player_ = 0;
    } else if(round_==1&&private_bucket_post_dealt_ < num_players_){ //the dealer updates buckets independently according to the transition matrix
      SetPrivate(private_bucket_post_dealt_, move);
      // Move to player 1 (the Big Blind, who acts first postflop).
      if (private_bucket_post_dealt_ == num_players_) cur_player_ = 1;
    }else SpielFatalError("Round number too large in ChanceNode in DoApplyAction");
  } else {
    SequenceAppendMove(move);
    //Here we encode the action in base 5. 1st digit is action type, 2nd digit is bet sizing index (0 for kF, kX, kC, 0-2 for kB, 0-1 for kR)
    int move_type = move%5;
    int bet_ind = move/5;
    if(move_type == ActionType::kF){
      // Player is now out.
      auto it = std::lower_bound(players_remaining_.begin(), players_remaining_.end(), cur_player_);
      if(it == players_remaining_.end()) SpielFatalError("Couldn't find current player in DoApplyAction.");
      players_remaining_.erase(it);
      action_is_closed_ = true;
      ResolveWinner(); //2 player game - when one folds, it ends;
    }else if(move_type==ActionType::kX){ //checking - just move the game along
      SPIEL_CHECK_EQ(cur_max_bet_, cur_round_bet_[cur_player_]);
      if(round_==0) SPIEL_CHECK_NE(cur_player_, 0); //The button cannot check postflop
      if(cur_player_!=last_to_act_) cur_player_ = NextPlayer();
      else{
        action_is_closed_ = true;
        if(!IsTerminal()) NewRound();
        else ResolveWinner();
      }
    }else if(move_type == ActionType::kC){
      SPIEL_CHECK_GE(cur_max_bet_, 1+cur_round_bet_[cur_player_]);
      int to_call = cur_max_bet_-cur_round_bet_[cur_player_];
      SPIEL_CHECK_GE(stack_[cur_player_], to_call);
      stack_[cur_player_] -= to_call;
      ante_[cur_player_] += to_call;
      cur_round_bet_[cur_player_] += to_call;
      pot_ += to_call;
      action_is_closed_ = true;
      if(round_==0&&cur_player_==0&&cur_max_bet_<=kBlinds[1]) action_is_closed_ = false;

      if (IsTerminal()) ResolveWinner();
      else if(action_is_closed_) NewRound();
      else cur_player_ = NextPlayer();
    }else if(move_type == ActionType::kB){
      SPIEL_CHECK_EQ(cur_max_bet_, 0);
      int to_bet = (int) (bet_sizes[bet_ind]*pot_);
      to_bet = std::max(to_bet, kBlinds[1]); //min bet rule: cannot bet less than 1BB
      to_bet = std::min(to_bet, stack_[cur_player_]); //cannot bet more than stack, takes priority over the min bet rule
      stack_[cur_player_] -= to_bet;
      ante_[cur_player_] += to_bet;
      cur_max_bet_ += to_bet;
      cur_round_bet_[cur_player_] += to_bet;
      pot_ += to_bet;
      nr_raises_++;

      if (IsTerminal()) SpielFatalError("Cannot be terminal after a bet");
      else cur_player_ = NextPlayer();
    }else if(move_type==ActionType::kR){
      int to_raise = (int)(cur_max_bet_-cur_round_bet_[cur_player_]+raise_sizes[bet_ind]*(pot_+cur_max_bet_-cur_round_bet_[cur_player_]));
      if(raise_sizes[bet_ind]*(pot_+cur_max_bet_-cur_round_bet_[cur_player_])<cur_max_bet_-cur_round_bet_[cur_player_]) to_raise = 2*(cur_max_bet_-cur_round_bet_[cur_player_]); //min raise rule - must be at least equal to the previous raise
      to_raise = std::min(to_raise, stack_[cur_player_]); //cannot raise more than stack
      stack_[cur_player_] -= to_raise;
      ante_[cur_player_] += to_raise;
      cur_round_bet_[cur_player_] += to_raise;
      cur_max_bet_ = cur_round_bet_[cur_player_];
      pot_ += to_raise;
      nr_raises_++;

      if (IsTerminal()) SpielFatalError("Cannot be terminal after a raise");
      else cur_player_ = NextPlayer();
    }else SpielFatalError(absl::StrCat("Move ", move, " is invalid. ChanceNode?", IsChanceNode()));
  }
  
  //std::cout << "Stacks: [" << stack_[0] << ", " << stack_[1] << "], Pot: " << pot_ << std::endl; 
  /*
  for(auto sclass:suit_classes_){
    std::cout << "{";
    for(int suit:sclass) std::cout << suit << ", ";
    std::cout << "}, ";
  }
  std::cout << std::endl;
  */
}

std::vector<std::vector<Card>> PostflopState::GetIso(std::vector<std::vector<Card>> classes) const{
  //create map from suit class -> counter
  std::map<std::vector<int>, int> class_counter;
  for(auto sclass:suit_classes_) class_counter[sclass] = 0;
  //permutate suits
  for(auto& sclass: classes){
    if(sclass.empty()) continue;
    //find the suit in suit_classes_
    for(int i=0;i<(int)suit_classes_.size();i++){
      if(std::find(suit_classes_[i].begin(), suit_classes_[i].end(), sclass[0].suit)!=suit_classes_[i].end()){
        //convert every card in this class to the suit indicated by suit_classes_[class_counter]
        for(Card& c:sclass) c.suit = suit_classes_[i][class_counter[suit_classes_[i]]];
        class_counter[suit_classes_[i]]++;
        break;
      }
    }
  }
  return classes;
}

int PostflopState::GetCardIndex(Card c) const{
  int nr_suits = default_deck_size/13;
  return nr_suits*c.rank+c.suit;
}

Hole HoleFromString(std::string s){
  std::vector<Card> hole;
  for(int i=0;i<s.length();i++){
    if(s[i]=='-'){
      int j=i-1;
      for(;j>=0;j--) if('0'>s[j]||s[j]>'9') break;
      int rank = std::stoi(s.substr(j+1, i-1-j));
      int suit = s[i+1]-'0';
      hole.push_back(Card(rank, suit));
    }
  }
  return Hole(hole);
}

std::vector<std::vector<Card>> PostflopState::GetClasses(std::vector<int> inds, bool to_sort) const{
  std::vector<std::vector<Card>> classes(4);
  for(int ind:inds) classes[deck_[ind].suit].push_back(deck_[ind]);
  if(to_sort) std::sort(classes.begin(), classes.end(), comp);
  return classes;
}

void PostflopState::UpdateSuitClasses(std::vector<Card> cs, std::vector<std::vector<int>> my_suit_classes){
  bool equiv[4][4];
  for(int i=0;i<4;i++) for(int j=0;j<4;j++) equiv[i][j] = false;
  for(auto sclass:my_suit_classes){
    for(int i=0;i<(int)sclass.size();i++){
      for(int j=i;j<(int)sclass.size();j++){
        equiv[sclass[i]][sclass[j]] = true;
        equiv[sclass[j]][sclass[i]] = true;
      }
    }
  }

  std::vector<std::vector<Card>> classes(4);
  for(Card c:cs) classes[c.suit].push_back(c);
  
  for(int i=0;i<4;i++){
    for(int j=i+1;j<4;j++){
      //std::cout << "equiv[" << i << "][" << j << "] = " << equiv[i][j] << " comp_eq: " << comp_eq(classes[i], classes[j]) << std::endl;
      equiv[i][j] &= comp_eq(classes[i], classes[j]);
      equiv[j][i] = equiv[i][j];
    }
  }
  for(int i=0;i<4;i++){
    for(int j=0;j<4;j++){
      for(int k=0;k<4;k++){
        if(equiv[i][j]&&equiv[j][k]&&!equiv[i][k]) SpielFatalError("Suit equivalence is not transitive");
      }
    }
  }
  suit_classes_.clear();
  std::vector<bool> visited(4, false);
  for(int i=0;i<4;i++){
    if(visited[i]) continue;
    std::vector<int> cur_class;
    for(int j=i;j<4;j++){
      if(equiv[i][j]){
        visited[j] = true;
        cur_class.push_back(j);
      }
    }
    suit_classes_.push_back(cur_class);
  }
}

std::vector<Action> PostflopState::LegalActions() const {
  if (IsTerminal()) return {};
  std::vector<Action> movelist;
  if (IsChanceNode()) {
    movelist.resize(num_buckets_);
    if(round_==0){ //every bucket is possible
      std::iota(movelist.begin(), movelist.end(), 0);
    }else if(round_==1){
      std::iota(movelist.begin(), movelist.end(), 0);
    }else SpielFatalError("round_ too big in LegalActions");
    return movelist;
  }

  // Can't just randomly fold; only allow fold when under pressure.
  if (cur_max_bet_>cur_round_bet_[cur_player_]){
    movelist.push_back(ActionType::kF);
    movelist.push_back(ActionType::kC);
  }
  // Can only chek if the current bet is 0, or if we are the big blind postflop after a limp
  if (cur_max_bet_==cur_round_bet_[cur_player_]) movelist.push_back(ActionType::kX);
  if(nr_raises_<max_raises_){ //Cannot bet or raise if already reached the max in the current round
    //Can bet postflop if the current bet is 0 and the stack allows
    if(cur_max_bet_==0&&round_>0&&stack_[cur_player_]>0){
      for(int i=0;i<(int)bet_sizes.size();i++){
        movelist.push_back(ActionType::kB+5*i);
        int to_bet = (int) (bet_sizes[i]*pot_);
        if(to_bet>stack_[cur_player_]) break;
      }
    }
    //Can raise if we are postflop, or if cur_max_bet_>cur_round_bet_[cur_player_], and the stack allows
    if((round_==0||cur_max_bet_>cur_round_bet_[cur_player_])&&stack_[cur_player_]>cur_max_bet_-cur_round_bet_[cur_player_]){
      for(int i=0;i<(int)raise_sizes.size();i++){
        movelist.push_back(ActionType::kR+5*i);
        int to_raise = (int)(cur_max_bet_-cur_round_bet_[cur_player_]+raise_sizes[i]*(pot_+cur_max_bet_-cur_round_bet_[cur_player_]));
        if(to_raise>stack_[cur_player_]) break;
      }
    }
  }
  return movelist;
}

std::string PostflopState::ActionToString(Player player, Action move) const {
  return GetGame()->ActionToString(player, move);
}

std::string PostflopState::ToString() const {
  std::string result;

  absl::StrAppend(&result, "Round: ", round_, "\nPlayer: ", cur_player_,
                  "\nPot: ", pot_, "\nMoney (p1 p2 ...):");
  for (auto p = Player{0}; p < num_players_; p++) {
    absl::StrAppend(&result, " ", stack_[p]);
  }
  absl::StrAppend(&result, "\nPre: ");
  for (Player player_index = 0; player_index < num_players_; player_index++) {
    absl::StrAppend(&result, private_bucket_pre_[player_index], " ");
  }
  absl::StrAppend(&result, "\nPost: ");
  for (Player player_index = 0; player_index < num_players_; player_index++) {
    absl::StrAppend(&result, private_bucket_post_[player_index], " ");
  }
  absl::StrAppend(&result, "\nRound 0 sequence: ");
  for (int i = 0; i < round0_sequence_.size(); ++i) {
    Action action = round0_sequence_[i];
    if (i > 0) absl::StrAppend(&result, ", ");
    absl::StrAppend(&result, StatelessActionToString(action));
  }
  absl::StrAppend(&result, "\nRound 1 sequence: ");
  for (int i = 0; i < round1_sequence_.size(); ++i) {
    Action action = round1_sequence_[i];
    if (i > 0) absl::StrAppend(&result, ", ");
    absl::StrAppend(&result, StatelessActionToString(action));
  }
  absl::StrAppend(&result, "\n");

  return result;
}

bool PostflopState::IsTerminal() const {
  int final_round = 1;
  return (int)players_remaining_.size() == 1 || (round_ == final_round && action_is_closed_);
}

std::vector<double> PostflopState::Returns() const {
  if (!IsTerminal()) {
    return std::vector<double>(num_players_, 0.0);
  }

  std::vector<double> returns(num_players_);
  //std::cout << "Returns: [";
  for (auto player = Player{0}; player < num_players_; ++player) {
    // Money vs money at start.
    returns[player] = stack_[player] - kDefaultStacks;
    //std::cout << returns[player] << (player==num_players_-1?"]":", ");
  }
  //std::cout<<std::endl;
  return returns;
}

// Information state is card then bets.
std::string PostflopState::InformationStateString(Player player) const {
  const PostflopGame& game = open_spiel::down_cast<const PostflopGame&>(*game_);
  return game.info_state_observer_->StringFrom(*this, player);
}

// Observation is card then contribution of each players to the pot.
std::string PostflopState::ObservationString(Player player) const {
  const PostflopGame& game = open_spiel::down_cast<const PostflopGame&>(*game_);
  return game.default_observer_->StringFrom(*this, player);
}

void PostflopState::InformationStateTensor(Player player,
                                        absl::Span<float> values) const {
  ContiguousAllocator allocator(values);
  const PostflopGame& game = open_spiel::down_cast<const PostflopGame&>(*game_);
  game.info_state_observer_->WriteTensor(*this, player, &allocator);
}

void PostflopState::ObservationTensor(Player player,
                                   absl::Span<float> values) const {
  ContiguousAllocator allocator(values);
  const PostflopGame& game = open_spiel::down_cast<const PostflopGame&>(*game_);
  game.default_observer_->WriteTensor(*this, player, &allocator);
}

std::unique_ptr<State> PostflopState::Clone() const {
  return std::unique_ptr<State>(new PostflopState(*this));
}

std::vector<std::pair<Hole, double>> PostflopState::ComputeHands() const{
  std::vector<std::pair<Hole, double>> holes;
  std::map<Hole, double> outcome_map;
  for(int i=0;i<default_deck_size;i++){
    if(deck_[i] == kInvalidCard) continue;
    for(int j=i+1;j<default_deck_size;j++){
      if(deck_[j] == kInvalidCard) continue;
      for(int k=j+1;k<default_deck_size;k++){
        if(deck_[k] == kInvalidCard) continue;
        for(int l=k+1;l<default_deck_size;l++){
          if(deck_[l] == kInvalidCard) continue;
          std::vector<std::vector<Card>> classes = GetClasses({i,j,k,l});
          std::vector<std::vector<Card>> iso = GetIso(classes);
          std::vector<Card> iso_flattened;
          for(auto sclass:iso) for(Card c:sclass) iso_flattened.push_back(c);
          Hole this_hole = Hole(iso_flattened);
          if(outcome_map.find(this_hole)==outcome_map.end()) outcome_map[this_hole] = 1.0;
          else outcome_map[this_hole] += 1.0;
        }
      }
    }
  }
  double sum_outcomes = 0;
  for(auto outc:outcome_map) sum_outcomes += outc.second;
  for(auto& outc:outcome_map) outc.second/=sum_outcomes;
  for(auto outc:outcome_map) holes.push_back({outc.first, outc.second});
  return holes;
}

std::vector<std::pair<Action, double>> PostflopState::ChanceOutcomes() const {
  SPIEL_CHECK_TRUE(IsChanceNode());
  std::vector<std::pair<Action, double>> outcomes;
  if(round_==0){ //deal buckets to each player
    for(int i=0;i<num_buckets_;i++) outcomes.push_back({i, postflop_prob_[i]});
  }else if(round_==1){ //the current player we're dealing with is private_bucket_post_dealt_. We retrieve its postflop bucket, and return the corresponding row in the transition matrix
    for(int i=0;i<num_buckets_;i++) outcomes.push_back({i, transition_matrix_[private_bucket_pre_[private_bucket_post_dealt_]][i]});
  }else{
    SpielFatalError("round number too large in ChanceOutcomes");
  }
  return outcomes;
}

int PostflopState::NextPlayer() const {
  // If we are on a chance node, it is the first player to play
  int nr_rounds = 2;
  if((round_==0&&private_bucket_pre_dealt_<num_players_)||(round_==1&&private_bucket_post_dealt_<num_players_)) {
    return kChancePlayerId;
  }
  if(cur_player_==kChancePlayerId){
    return (int)(round_>0); //trick: postflop (0) the button (0) acts first, postflop (1) the big blind (1) acts first;
  }
  auto it = std::lower_bound(players_remaining_.begin(), players_remaining_.end(), cur_player_);
  if(it == players_remaining_.end()) SpielFatalError("Could not find player in NextPlayer.");
  if((++it)==players_remaining_.end()) it = players_remaining_.begin();
  return *it;
}

HandScore PostflopState::GetScoreFrom5(std::vector<Card> cards) const {
  //Returns the hand score of a set of 5 cards. First comes the absolute rank (https://en.wikipedia.org/wiki/List_of_poker_hands#Hand-ranking_categories)
  //Then come the revelant "kicker" information - how to dispute ties in case of same hand rank
  //A HandScore is better iff its vector is lexicographically greater. See implementation in postflop.h
  std::sort(cards.begin(), cards.end());
  if(std::find(cards.begin(), cards.end(), kInvalidCard)!=cards.end()) SpielFatalError("Trying to score kInvalidCard");
  bool flush = true;
  bool straight = true;
  bool ato5straight = false;
  int trips = -1;
  int pairs = 0;
  int pair_ind = -1;
  //check for straights and flushes
  for(int i=0;i<5;i++){
    if(cards[i].suit!=cards[0].suit) flush = false;
    if(i>0&&cards[i].rank!=cards[i-1].rank+1) straight = false;
    if(i>1&&cards[i].rank==cards[i-1].rank&&cards[i-1].rank==cards[i-2].rank) trips = i;
    if(i>0&&cards[i].rank==cards[i-1].rank){
      pairs++;
      pair_ind = i;
    }
  }
  //check for A-5 straight
  if(cards[0].rank==0&&cards[1].rank==1&&cards[2].rank==2&&cards[3].rank==3&&cards[4].rank==12) ato5straight = true;
  std::vector<int> hand_class(6, -1);
  if((straight||ato5straight)&&flush){ // 9 = straight flush
    hand_class[0] = 9;
    if(straight) hand_class[1] = cards[4].rank;
    else hand_class[1] = cards[3].rank; // in A to 5 straight, highest card is the 5
  }else if(cards[0].rank==cards[3].rank||cards[1].rank==cards[4].rank){ //8 = four of a kind
    hand_class[0] = 8;
    hand_class[1] = cards[2].rank; // get the four of a kind card
    //get the kicker
    if(cards[0].rank==cards[3].rank) hand_class[2] = cards[4].rank;
    else hand_class[2] = cards[0].rank;
  }else if((cards[0].rank==cards[2].rank&&cards[3].rank==cards[4].rank)||(cards[0].rank==cards[1].rank&&cards[2].rank==cards[4].rank)){ // 7 = full house
    hand_class[0] = 7;
    hand_class[1] = cards[2].rank; // get the three of a kind card
    //get the kicker (pair)
    if(cards[0].rank==cards[2].rank&&cards[3].rank==cards[4].rank) hand_class[2] = cards[3].rank;
    else hand_class[2] = cards[0].rank;
  }else if(flush){ // 6 = flush
    hand_class[0] = 6;
    for(int i=4;i>=0;i--) hand_class[5-i] = cards[i].rank; // all cards could be necessary to dispute ties
  }else if(straight||ato5straight){ // 5 = straight
    hand_class[0] = 5;
    if(straight) hand_class[1] = cards[4].rank;
    else hand_class[1] = cards[3].rank; // in A to 5 straight, highest card is the 5
  }else if(trips>=0){ // 4 = three of a kind
    hand_class[0] = 4;
    hand_class[1] = cards[trips].rank;
    int ind_lo = (trips+1)%4, ind_hi = (trips+2)%5;
    if(ind_lo>ind_hi) std::swap(ind_lo, ind_hi);
    hand_class[2] = cards[ind_hi].rank;
    hand_class[3] = cards[ind_lo].rank;
  }else if(pairs>=2){ // 3 = two pairs
    hand_class[0] = 3;
    if(cards[0].rank==cards[1].rank){
      if(cards[2].rank==cards[3].rank){
        hand_class[1] = cards[3].rank; // hi pair
        hand_class[2] = cards[1].rank; // lo pair
        hand_class[3] = cards[4].rank; // kicker
      }else{
        hand_class[1] = cards[4].rank; // hi pair
        hand_class[2] = cards[1].rank; // lo pair
        hand_class[3] = cards[2].rank; // kicker
      }
    }else{
      hand_class[1] = cards[4].rank; // hi pair
      hand_class[2] = cards[2].rank; // lo pair
      hand_class[3] = cards[0].rank; // kicker
    }
  }else if(pairs>0){ // 2 = one pair
    hand_class[0] = 2;
    hand_class[1] = cards[pair_ind].rank;
    std::vector<int> other_inds = {(pair_ind+1)%5, (pair_ind+2)%5, (pair_ind+3)%5};
    std::sort(other_inds.begin(), other_inds.end());
    for(int i=0;i<3;i++) hand_class[2+i] = cards[other_inds[2-i]].rank;
  }else{ // 1 = high card
    hand_class[0] = 1;
    for(int i=4;i>=0;i--) hand_class[5-i] = cards[i].rank; // all cards could be necessary to dispute ties
  }
  return HandScore(hand_class);
}

HandScore PostflopState::RankHand(Player player) const {
  HandScore max_score = HandScore({-1, -1, -1, -1, -1, -1});
  /*
  for(int i=0;i<(int)public_cards_.size();i++){
    for(int j=i+1;j<(int)public_cards_.size();j++){
      for(int k=j+1;k<(int)public_cards_.size();k++){
        for(int l=0;l<4;l++){
          for(int m=l+1;m<4;m++){
            max_score = std::max(max_score, GetScoreFrom5({private_hole_[player].cards[l], private_hole_[player].cards[m], public_cards_[i], public_cards_[j], public_cards_[k]}));
          }
        }
      }
    }
  }
  SPIEL_CHECK_NE(max_score.score[0], -1);
  */
  return max_score;
}

void PostflopState::ResolveWinner() {
  num_winners_ = kInvalidPlayer;

  if ((int)players_remaining_.size() == 1) {
    winner_[players_remaining_[0]] = true;
    num_winners_ = 1;
    stack_[players_remaining_[0]] += pot_; // += (1-rake_)*pot_, if raked game
    pot_ = 0;
    return;
  } else {
    // Otherwise, showdown!
    // Many possible algorithms for comparing buckets. We award a fraction of the pot proportional to the equity squared (gives more weight to good buckets)
    double sum_eqsq = 0;
    double bucket_len = 1.0/num_buckets_;
    for(Player p:players_remaining_) sum_eqsq += (bucket_len*(double)private_bucket_post_[p]+bucket_len/2)*(bucket_len*(double)private_bucket_post_[p]+bucket_len/2);
    int tot = 0;
    for(Player p:players_remaining_){
      int removed = static_cast<double>(pot_) * (bucket_len*(double)private_bucket_post_[p]+bucket_len/2)*(bucket_len*(double)private_bucket_post_[p]+bucket_len/2)/sum_eqsq;
      stack_[p] += removed;
      tot += removed;
    }
    pot_ -= tot;
    //Deal with rounding errors
    stack_[players_remaining_[0]] += pot_;
    pot_ = 0;
  }
}

void PostflopState::NewRound() {
  round_++;
  cur_player_ = kChancePlayerId;
  last_to_act_ = 0;
  cur_max_bet_ = 0;
  nr_raises_ = 0;
  action_is_closed_ = false;
  std::fill(cur_round_bet_.begin(), cur_round_bet_.end(), 0);
}

void PostflopState::SequenceAppendMove(int move) {
  if (round_ == 0) {
    round0_sequence_.push_back(move);
  } else if(round_==1){
    round1_sequence_.push_back(move);
  }else SpielFatalError("SequenceAppendMove: round has to be in [0, 1]");
}

std::vector<int> PostflopState::padded_betting_sequence() const {
  std::vector<int> history = round0_sequence_;

  // We pad the history to the end of the first round with kPaddingAction.
  history.resize(game_->MaxGameLength() / 2, kInvalidAction);

  // We insert the actions that happened in the second round, and fill to
  // MaxGameLength.
  history.insert(history.end(), round1_sequence_.begin(),
                 round1_sequence_.end());
  history.resize(game_->MaxGameLength(), kInvalidAction);
  return history;
}

void PostflopState::SetPrivate(Player player, Action move) {
	if(round_==0){
    private_bucket_pre_[player] = move;
    ++private_bucket_pre_dealt_;
  }else if(round_==1){
    private_bucket_post_[player] = move;
    ++private_bucket_post_dealt_;
  }else SpielFatalError("Round not 0 or 1 in SetPrivate");
}

std::unique_ptr<State> PostflopState::ResampleFromInfostate(
    int player_id, std::function<double()> rng) const {
  std::unique_ptr<State> clone = game_->NewInitialState();

  // First, deal out cards:
  Action player_chance = history_.at(player_id).action;
  for (int p = 0; p < GetGame()->NumPlayers(); ++p) {
    if (p == player_id) {
      clone->ApplyAction(history_.at(p).action);
    } else {
      Action chosen_action = player_chance;
      while (chosen_action == player_chance) {
        chosen_action = SampleAction(clone->ChanceOutcomes(), rng()).first;
      }
      clone->ApplyAction(chosen_action);
    }
  }
  for (int action : round0_sequence_) clone->ApplyAction(action);
  for (int action : round1_sequence_) clone->ApplyAction(action);
  return clone;
}

void PostflopState::SetPrivateBuckets(const std::vector<int>& new_private_bucket) {
  SPIEL_CHECK_EQ(new_private_bucket.size(), NumPlayers());
  private_bucket_pre_ = new_private_bucket;
  private_bucket_post_ = new_private_bucket;
}

PostflopGame::PostflopGame(const GameParameters& params)
    : Game(kGameType, params),
      num_players_(ParameterValue<int>("players")),
      num_buckets_(ParameterValue<int>("num_buckets")){
  SPIEL_CHECK_GE(num_players_, kGameType.min_num_players);
  SPIEL_CHECK_LE(num_players_, kGameType.max_num_players);
  default_observer_ = std::make_shared<PostflopObserver>(kDefaultObsType);
  info_state_observer_ = std::make_shared<PostflopObserver>(kInfoStateObsType);
}

std::unique_ptr<State> PostflopGame::NewInitialState() const {
  return absl::make_unique<PostflopState>(shared_from_this(), num_buckets_);
}

int PostflopGame::MaxChanceOutcomes() const {
  return 1000000000;
}

std::vector<int> PostflopGame::InformationStateTensorShape() const {
  // One-hot encoding for player number (who is to play).
  // 2 slots of cards (PLACEHOLDER bits each): private card, public card
  // Followed by maximum game length * 2 bits each (call / raise)
  if (true) {
    return {(num_players_) + (PLACEHOLDER) + (100 * 2)};
  } else {
    return {(num_players_) + (PLACEHOLDER * 2) + (100 * 2)};
  }
}

std::vector<int> PostflopGame::ObservationTensorShape() const {
  // One-hot encoding for player number (who is to play).
  // 2 slots of cards (PLACEHOLDER bits each): private card, public card
  // Followed by the contribution of each player to the pot
  if (true) {
    return {(num_players_) + (PLACEHOLDER) + (num_players_)};
  } else {
    return {(num_players_) + (PLACEHOLDER * 2) + (num_players_)};
  }
}

double PostflopGame::MaxUtility() const {
  // In poker, the utility is defined as the money a player has at the end of
  // the game minus then money the player had before starting the game.
  // The most a player can win *per opponent* is the most each player can put
  // into the pot, which is the raise amounts on each round times the maximum
  // number raises, plus the original chip they put in to play.
  return (num_players_ - 1);
}

double PostflopGame::MinUtility() const {
  // In poker, the utility is defined as the money a player has at the end of
  // the game minus then money the player had before starting the game.
  // The most any single player can lose is the maximum number of raises per
  // round times the amounts of each of the raises, plus the original chip
  // they put in to play.
  return -1;
}

std::shared_ptr<Observer> PostflopGame::MakeObserver(
    absl::optional<IIGObservationType> iig_obs_type,
    const GameParameters& params) const {
  if (params.empty()) {
    return std::make_shared<PostflopObserver>(
        iig_obs_type.value_or(kDefaultObsType));
  } else {
    return MakeRegisteredObserver(iig_obs_type, params);
  }
}

std::string PostflopGame::ActionToString(Player player, Action action) const {
  if (player == kChancePlayerId) {
    return absl::StrCat("Chance outcome:", action);
  } else {
    return StatelessActionToString(action);
  }
}

}  // namespace postflop
}  // namespace open_spiel
