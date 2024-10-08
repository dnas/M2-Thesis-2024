#include<iostream>
#include<algorithm>
#include<cstdio>
#include<vector>
#include<cmath>
#include<random>
#include<bitset>
#include<string>
#include<queue>
#include<stack>
#include<set>
#include<map>
#include<deque>
#include<chrono>
#include<fstream>

std::mt19937 rng(5334);

class Card{
  public:
    int rank, suit;
    Card(int r, int s){
      rank = r; suit = s;
    }
    Card(std::pair<int, int> rs){
      rank = rs.first; suit = rs.second;
    }
    Card(std::vector<int> rs){
      rank = rs[0]; suit = rs[1];
    }
    bool operator <(const Card& card2) const{
			return rank==card2.rank?suit<card2.suit:rank<card2.rank;
    }
		bool operator ==(const Card& card2) const{
			return rank==card2.rank&&suit==card2.suit;
    }
		bool operator !=(const Card& card2) const{
			return !(*this==card2);
    }
		std::string ToString() const{
			return std::to_string(rank)+"-"+std::to_string(suit);
		}

    friend std::ostream& operator<< (std::ostream& stream, const Card& card) {
      stream << card.ToString();
      return stream;
    }
};

inline const Card kInvalidCard{-10000, -10000};

bool comp(std::vector<Card> a, std::vector<Card> b){
  std::sort(a.begin(), a.end()); std::sort(b.begin(), b.end());
  int i = 0;
  while(i<(int)a.size()&&i<(int)b.size()&&a[i].rank==b[i].rank) i++;
  if(i<(int)a.size()&&i<(int)b.size()) return a[i].rank<b[i].rank;
  return a.size()<b.size();
}

bool comp_eq(std::vector<Card> a, std::vector<Card> b){
  std::sort(a.begin(), a.end()); std::sort(b.begin(), b.end());
  int i = 0;
  while(i<(int)a.size()&&i<(int)b.size()&&a[i].rank==b[i].rank) i++;
  if(i<(int)a.size()&&i<(int)b.size()) return false;
  return a.size()==b.size();
}

class Hole{
  public:
    std::vector<Card> cards;
    Hole(){
      cards.assign(4, kInvalidCard);
    }
    Hole(Card c0, Card c1, Card c2, Card c3){
      cards = {c0, c1, c2, c3};
    }
    Hole(std::vector<Card> cs){
      cards = cs;
    }
		bool operator ==(const Hole& hole2) const{
			return cards==hole2.cards;
    }
		bool operator !=(const Hole& hole2) const{
			return !(*this==hole2);
    }
    bool operator <(const Hole& hole2) const{
      for(int i=0;i<4;i++) if(cards[i]!=hole2.cards[i]) return cards[i]<hole2.cards[i];
      return false;
    }
		std::string ToString() const{
			return "["+cards[0].ToString()+","+cards[1].ToString()+","+cards[2].ToString()+","+cards[3].ToString()+"]";
		}
		void sort(){
			std::sort(cards.begin(), cards.end());
		}
};

class HandScore{
  public:
    std::vector<int> score;
    HandScore(std::vector<int> sc){
      score = sc;
    }
    bool operator ==(const HandScore& hsc2) const{
			return score==hsc2.score;
    }
    bool operator <(const HandScore& hsc2) const{
      for(int i=0;i<(int)score.size();i++){
        if(score[i]==hsc2.score[i]) continue;
        return score[i]<hsc2.score[i];
      }
      return false;
    }
};

// Default parameters.
inline const Hole kInvalidHole{kInvalidCard, kInvalidCard, kInvalidCard, kInvalidCard};
const int default_deck_size = 52;

//Public information
std::vector<Card> public_cards_;  // The public card revealed after round 1.
//Player information
std::vector<Hole> private_hole_;
// Cards by value (0-6 for standard 2-player game, -1 if no longer in the
// deck.)
std::vector<Card> deck_;

std::vector<std::vector<int>> suit_classes_; //Current vector of classes of indistinguishable suits, plus a counter for each class.
//for example, we might have the pairs {{0,3}, {1,2}}, meaning that suits 0,3 are equivalent, and same for 1,2.
std::vector<std::vector<int>> suit_classes_flop_; //A snapshot of the above vector just before the flop is dealt.
//This is important because the turn may join new classes again

int IndexFromCard(Card c){
  return 4*c.rank+c.suit;
}

Card CardFromIndex(int ind){
  return Card(ind/4, ind%4);
}

std::vector<std::vector<Card>> GetClasses(std::vector<int> inds, bool to_sort = true){
  std::vector<std::vector<Card>> classes(4);
  for(int ind:inds) classes[deck_[ind].suit].push_back(deck_[ind]);
  if(to_sort) std::sort(classes.begin(), classes.end(), comp);
  return classes;
}

void UpdateSuitClasses(std::vector<Card> cs, std::vector<std::vector<int>> my_suit_classes){
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
        if(equiv[i][j]&&equiv[j][k]&&!equiv[i][k]){
          std::cerr << "Suit classes are non-transitive" << std::endl;
          exit(1);
        }
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

std::vector<std::vector<Card>> GetIso(std::vector<std::vector<Card>> classes){
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

HandScore GetScoreFrom5(std::vector<Card> cards){
  //Returns the hand score of a set of 5 cards. First comes the absolute rank (https://en.wikipedia.org/wiki/List_of_poker_hands#Hand-ranking_categories)
  //Then come the revelant "kicker" information - how to dispute ties in case of same hand rank
  //A HandScore is better (wins) iff its vector is lexicographically greater. See implementation in plo.h
  std::sort(cards.begin(), cards.end());
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

HandScore RankHand(int player){
  HandScore max_score = HandScore({-1, -1, -1, -1, -1, -1});
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
  if(max_score.score[0] == -1){
    std::cerr << "max_score is still -1" << std::endl;
    exit(1);
  }
  return max_score;
}

std::vector<std::pair<Hole, double>> ComputeHands(){
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

double RolloutEquity(int nr_rollouts){
  std::vector<Card> slim_deck;
  for(int i=0;i<default_deck_size;i++) if(deck_[i]!=kInvalidCard) slim_deck.push_back(deck_[i]);
  double equity = 0;
  for(int iii=0;iii<nr_rollouts;iii++){
    std::vector<int> inds_changed(5, -1);
    for(int i=0;i<5;i++){
      while(public_cards_[i]==kInvalidCard){
        int rnd_ind = rng()%((int) slim_deck.size());
        public_cards_[i] = slim_deck[rnd_ind];
        slim_deck[rnd_ind] = kInvalidCard;
        inds_changed[i] = rnd_ind;
      }
    }
    HandScore p0 = RankHand(0); HandScore p1 = RankHand(1);
    if(p1<p0) equity += 1.0;
    else if(p1==p0) equity += 0.5;
    for(int i=0;i<5;i++) slim_deck[inds_changed[i]] = public_cards_[i];
    public_cards_.assign(5, kInvalidCard);
  }
  return equity/nr_rollouts;
}

void AllRolloutEquity(int nr_opphands, int nr_rollouts){
  //Estimates E[HS] (https://poker.cs.ualberta.ca/publications/AAMAS13-abstraction.pdf)
  //using nr_opphands as the number of uniformly sampled opponent hands, and nr_rollouts as the number of rollouts
  std::vector<std::pair<Hole, double>> all_hands = ComputeHands();
  std::reverse(all_hands.begin(), all_hands.end());
  for(auto pair_hd:all_hands){
    Hole cur_hole = pair_hd.first;
    //Update information
    for(int i=0;i<4;i++) deck_[IndexFromCard(cur_hole.cards[i])] = kInvalidCard;
    UpdateSuitClasses(cur_hole.cards, suit_classes_);
    private_hole_[0] = cur_hole;
    //Get rollout equity for this hand
    std::vector<std::pair<Hole, double>> opp_hands = ComputeHands();
    nr_opphands = std::min(nr_opphands, (int)opp_hands.size());
    std::shuffle(opp_hands.begin(), opp_hands.end(), rng);
    double total_equity = 0; double total_prob = 0;
    for(int iii=0;iii<nr_opphands;iii++){
      Hole opp_hole = opp_hands[iii].first;
      for(int i=0;i<4;i++) deck_[IndexFromCard(opp_hole.cards[i])] = kInvalidCard;
      private_hole_[1] = opp_hole;
      total_equity += RolloutEquity(nr_rollouts)*opp_hands[iii].second;
      total_prob += opp_hands[iii].second;

      for(int i=0;i<4;i++) deck_[IndexFromCard(opp_hole.cards[i])] = opp_hole.cards[i];
      private_hole_[1] = kInvalidHole;
    }
    total_equity/=total_prob; //normalization
    std::cout << cur_hole.ToString() << " " << total_equity << std::endl;
    //Go back to previous information
    for(int i=0;i<4;i++) deck_[IndexFromCard(cur_hole.cards[i])] = cur_hole.cards[i];
    suit_classes_.assign(1, {0,1,2,3});
    private_hole_[0] = kInvalidHole;
  }
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

bool CheckDuplicate(){
  std::set<Card> pcards;
  for(int i=0;i<5;i++) if(public_cards_[i]!=kInvalidCard) pcards.insert(public_cards_[i]);
  for(int i=0;i<4;i++) if(pcards.find(private_hole_[0].cards[i])!=pcards.end()) return true;
  return false;
}

double FlopEquity(int nr_rollouts){
  std::vector<Card> slim_deck;
  for(int i=0;i<default_deck_size;i++) if(deck_[i]!=kInvalidCard) slim_deck.push_back(deck_[i]);
  double equity = 0;
  for(int iii=0;iii<nr_rollouts;iii++){
    std::vector<int> inds_changed(2, -1);
    for(int i=3;i<5;i++){
      while(public_cards_[i]==kInvalidCard){
        int rnd_ind = rng()%((int) slim_deck.size());
        public_cards_[i] = slim_deck[rnd_ind];
        slim_deck[rnd_ind] = kInvalidCard;
        inds_changed[i-3] = rnd_ind;
      }
    }
    HandScore p0 = RankHand(0); HandScore p1 = RankHand(1);
    if(p1<p0) equity += 1.0;
    else if(p1==p0) equity += 0.5;
    for(int i=3;i<5;i++) slim_deck[inds_changed[i-3]] = public_cards_[i];
    for(int i=3;i<5;i++) public_cards_[i] = kInvalidCard;
  }
  return equity/nr_rollouts;
}

void AllFlopEquity(int nr_flops, int nr_opphands, int nr_rollouts){
  //Estimates E[HS] (https://poker.cs.ualberta.ca/publications/AAMAS13-abstraction.pdf) AFTER the flop was dealt
  //using nr_opphands as the number of uniformly sampled opponent hands, and nr_rollouts as the number of rollouts
  std::vector<std::pair<Hole, double>> all_hands = ComputeHands();
  std::vector<std::vector<int>> flop_inds;
  for(int iii=0;iii<nr_flops;){
    std::vector<int> cur_flop_ind;
    for(int i=0;i<3;i++) cur_flop_ind.push_back(rng()%default_deck_size);
    std::sort(cur_flop_ind.begin(), cur_flop_ind.end());
    if(cur_flop_ind[0]==cur_flop_ind[1]||cur_flop_ind[1]==cur_flop_ind[2]) continue;
    flop_inds.push_back(cur_flop_ind);
    iii++;
  }
  std::reverse(all_hands.begin(), all_hands.end());
  std::ofstream thefile;
  for(int i_flops = 0;i_flops<nr_flops;i_flops++){
    std::string file_name = "flops"+std::to_string(200+i_flops)+".txt";
    thefile.open(file_name);
    thefile << flop_inds[i_flops][0] << " " << flop_inds[i_flops][1] << " " << flop_inds[i_flops][2] << "\n";
    thefile << deck_[flop_inds[i_flops][0]].ToString() << " " << deck_[flop_inds[i_flops][1]].ToString() << " " << deck_[flop_inds[i_flops][2]].ToString() << "\n";
    for(int i=0;i<3;i++){
      public_cards_[i] = deck_[flop_inds[i_flops][i]];
      deck_[flop_inds[i_flops][i]] = kInvalidCard;
    }
    for(auto pair_hd:all_hands){
      Hole cur_hole = pair_hd.first;
      //Update information
      private_hole_[0] = cur_hole;
      if(CheckDuplicate()) continue;
      for(int i=0;i<4;i++) deck_[IndexFromCard(cur_hole.cards[i])] = kInvalidCard;
      //UpdateSuitClasses(cur_hole.cards, suit_classes_);
      //Get rollout equity for this hand
      std::vector<Card> slim_deck;
      for(int i=0;i<default_deck_size;i++) if(deck_[i]!=kInvalidCard) slim_deck.push_back(deck_[i]);
      double total_equity = 0;
      for(int i_opp=0;i_opp<nr_opphands;i_opp++){
        std::set<int> opp_cards_inds;
        while((int)opp_cards_inds.size()<4) opp_cards_inds.insert(rng()%((int) slim_deck.size()));
        auto it = opp_cards_inds.begin();
        Hole opp_hole;
        for(int i=0;i<4;i++){
          opp_hole.cards[i] = slim_deck[*it];
          deck_[IndexFromCard(slim_deck[*it])] = kInvalidCard;
          slim_deck[*it] = kInvalidCard;
          it++;
        }
        private_hole_[1] = opp_hole;
        total_equity += FlopEquity(nr_rollouts)*1.0/nr_opphands;
        for(int i=3;i>=0;i--){
          it--;
          std::swap(slim_deck[*it], opp_hole.cards[i]);
          deck_[IndexFromCard(slim_deck[*it])] = slim_deck[*it];
        }
        private_hole_[1] = kInvalidHole;
      }
      thefile << cur_hole.ToString() << " " << total_equity << std::endl;
      //Go back to previous information
      for(int i=0;i<4;i++) deck_[IndexFromCard(cur_hole.cards[i])] = cur_hole.cards[i];
      //suit_classes_.assign(1, {0,1,2,3});
      private_hole_[0] = kInvalidHole;
    }
    for(int i=0;i<3;i++) std::swap(public_cards_[i], deck_[flop_inds[i_flops][i]]);
    thefile.close();
  }
}

std::map<Hole, double> preflop_prob_;
std::map<Hole, double> preflop_strength_; //Given a hole card combination, what's it's rollout strength?
std::vector<std::pair<double, Hole>> strength_vector_; //same as above, in pair format
std::vector<std::vector<Hole>> buckets; //the k-th bucket contains all the holes with HS between 0.2+0.5*(k-1)/n and 0.2+0.5*k/n
std::vector<double> buckets_prob; //probability of being dealt each bucket
std::map<Hole, int> hole_to_bucket; //given a hole, which bucket it belongs to PREFLOP
std::map<Hole, std::vector<double>> transitioned; //for each hole, a vector of size nr_buckets which counts how many times that hole ended up in that hs bucket postflop
std::vector<std::vector<double>> transition_matrix; // bucket to bucket transition matrix (from preflop to flop)

void WritePreflop(int nr_buckets){
  std::ofstream ofile;
  std::string file_name = "preflop_buckets_prob_"+std::to_string(nr_buckets)+".txt";
  ofile.open(file_name);
  ofile << std::fixed << std::setprecision(10);
  for(int i=0;i<nr_buckets;i++){
    ofile << buckets_prob[i] << "\n";
  }
  ofile.close();
}

void WritePostflop(int nr_buckets){
  std::ofstream ofile;
  std::string file_name = "postflop_buckets_matrix_"+std::to_string(nr_buckets)+".txt";
  ofile.open(file_name);
  ofile << std::fixed << std::setprecision(10);
  for(int i=0;i<nr_buckets;i++){
    for(int j=0;j<nr_buckets;j++){
      ofile << transition_matrix[i][j] << " \n"[j+1==nr_buckets];
    }
  }
  ofile.close();
}

void ComputeBuckets(int nr_buckets, int nr_flops){
  std::vector<std::pair<Hole, double>> hole_and_prob = ComputeHands();
  for(auto x:hole_and_prob) preflop_prob_[x.first] = x.second;
  std::ifstream ifile;
  ifile.open("plo_preflop_hs.txt");
  std::string content;
  Hole temp_hole;
  double temp_strength;
  int zzz = 0;
  while(ifile >> content) {
    if(zzz%2==0) temp_hole = HoleFromString(content);
    else{
      temp_strength = stod(content);
      preflop_strength_[temp_hole] = temp_strength;
      strength_vector_.push_back({temp_strength, temp_hole});
    }
    zzz++;
  }
  ifile.close();
  //std::sort(strength_vector_.begin(), strength_vector_.end());
  double pre_l = 0.3, pre_r = 0.7;
  double int_length = (pre_r-pre_l)/nr_buckets;
  buckets.resize(nr_buckets);
  buckets_prob.assign(nr_buckets, 0.0);
  for(int i=0;i<(int)strength_vector_.size();i++){
    int cur_bucket = (int)((strength_vector_[i].first-pre_l)/int_length);
    cur_bucket = std::max(std::min(cur_bucket, nr_buckets-1), 0); //outliers are clipped to the extremes
    buckets[cur_bucket].push_back(strength_vector_[i].second);
    hole_to_bucket[strength_vector_[i].second] = cur_bucket;
    buckets_prob[cur_bucket] += preflop_prob_[strength_vector_[i].second];
  }
  WritePreflop(nr_buckets);
  double post_l = 0, post_r = 1.0;
  int_length = (post_r-post_l)/nr_buckets;
  for(int i_flops=0;i_flops<nr_flops;i_flops++){
    std::string file_name = "flops"+std::to_string(i_flops)+".txt";
    ifile.open(file_name);
    std::string content;
    zzz = -1;
    Hole cur_hole;
    double cur_strength;
    while(ifile >> content) {
      zzz++;
      if(zzz<6) continue;
      if(zzz%2==0) cur_hole = HoleFromString(content);
      else{
        cur_strength = stod(content);
        if(transitioned.find(cur_hole)==transitioned.end()) transitioned[cur_hole] = std::vector<double>(nr_buckets, 0.0);
        int cur_bucket = (cur_strength-post_l)/int_length;
        cur_bucket = std::max(std::min(cur_bucket, nr_buckets-1), 0); //outliers are clipped to the extremes
        transitioned[cur_hole][cur_bucket]+=1.0;
      }
    }
    ifile.close();
  }
  for(auto x:hole_and_prob){
    double tot_amt = 0;
    for(int i=0;i<nr_buckets;i++) tot_amt += transitioned[x.first][i];
    for(int i=0;i<nr_buckets;i++) transitioned[x.first][i] /= tot_amt;
  }
  transition_matrix.clear();
  for(int i=0;i<nr_buckets;i++){
    transition_matrix.push_back(std::vector<double>(nr_buckets, 0.0));
    for(Hole x:buckets[i]){
      for(int j=0;j<nr_buckets;j++){
        transition_matrix[i][j] += transitioned[x][j]*preflop_prob_[x]/buckets_prob[i];
      }
    }
  }
  WritePostflop(nr_buckets);
}

double TurnEquity(int nr_rivers){
  double equity = 0;
  for(int iii=0;iii<nr_rivers;iii++){
    std::vector<int> inds_changed(1, -1);
    for(int i=4;i<5;i++){
      while(public_cards_[i]==kInvalidCard){
        int rnd_ind = rng()%(default_deck_size);
        public_cards_[i] = deck_[rnd_ind];
        deck_[rnd_ind] = kInvalidCard;
        inds_changed[i-4] = rnd_ind;
      } 
    }
    HandScore p0 = RankHand(0); HandScore p1 = RankHand(1);
    if(p1<p0) equity += 1.0;
    else if(p1==p0) equity += 0.5;
    for(int i=4;i<5;i++) deck_[inds_changed[i-4]] = public_cards_[i];
    for(int i=4;i<5;i++) public_cards_[i] = kInvalidCard;
  }
  return equity/nr_rivers;
}

void TurnHS(std::vector<Card> flop, int nr_rivers, int nr_opphands){
  //Estimates E[HS] (https://poker.cs.ualberta.ca/publications/AAMAS13-abstraction.pdf) for a specific flop
  //using nr_opphands as the number of uniformly sampled opponent hands, nr_rivers as the number of rollouts per turn
  std::vector<std::pair<Hole, double>> all_hands;
  all_hands = ComputeHands();
  /*
  std::ifstream ifile;
  ifile.open("PLACEHOLDER.txt"); //should contain on each line: the hand in standard format, space, density in range
  std::string content;
  Hole temp_hole;
  double temp_density;
  int zzz = 0;
  while(ifile >> content) {
    if(zzz%2==0) temp_hole = HoleFromString(content);
    else{
      temp_density = stod(content);
      all_hands.push_back({temp_hole, temp_density});
    }
    zzz++;
  }
  ifile.close();
  */
  std::reverse(all_hands.begin(), all_hands.end());
  for(int i=0;i<3;i++){
    public_cards_[i] = flop[i];
    deck_[IndexFromCard(flop[i])] = kInvalidCard;
  }
  std::ofstream ofile;
  ofile.open("turn_hs.txt");
  for(int i_turn=0;i_turn<default_deck_size;i_turn++){
    if(deck_[i_turn]==kInvalidCard){
      continue;
    }
    std::swap(deck_[i_turn], public_cards_[3]);
    ofile << "Turn: " << public_cards_[3] << std::endl;
    for(auto pair_hd:all_hands){
      Hole cur_hole = pair_hd.first;
      //Update information
      private_hole_[0] = cur_hole;
      if(CheckDuplicate()) continue;
      for(int i=0;i<4;i++) deck_[IndexFromCard(cur_hole.cards[i])] = kInvalidCard;

      //Get rollout equity for this hand
      std::vector<Card> slim_deck;
      for(int i=0;i<default_deck_size;i++) if(deck_[i]!=kInvalidCard) slim_deck.push_back(deck_[i]);
      double total_equity = 0;
      for(int i_opp=0;i_opp<nr_opphands;i_opp++){
        std::set<int> opp_cards_inds;
        while((int)opp_cards_inds.size()<4) opp_cards_inds.insert(rng()%((int) slim_deck.size()));
        auto it = opp_cards_inds.begin();
        Hole opp_hole;
        for(int i=0;i<4;i++){
          opp_hole.cards[i] = slim_deck[*it];
          deck_[IndexFromCard(slim_deck[*it])] = kInvalidCard;
          slim_deck[*it] = kInvalidCard;
          it++;
        }
        private_hole_[1] = opp_hole;
        total_equity += TurnEquity(nr_rivers)*1.0/nr_opphands;
        for(int i=3;i>=0;i--){
          it--;
          std::swap(slim_deck[*it], opp_hole.cards[i]);
          deck_[IndexFromCard(slim_deck[*it])] = slim_deck[*it];
        }
        private_hole_[1] = kInvalidHole;
      }
      ofile << cur_hole.ToString() << " " << total_equity << std::endl;
      //Go back to previous information
      for(int i=0;i<4;i++) deck_[IndexFromCard(cur_hole.cards[i])] = cur_hole.cards[i];
      //suit_classes_.assign(1, {0,1,2,3});
      private_hole_[0] = kInvalidHole;
    }
    std::swap(deck_[i_turn], public_cards_[3]);
  }
  ofile.close();
}

int main(){
  //std::ios::sync_with_stdio(0); std::cin.tie(0); std::cout.tie(0);
	std::cout.precision(6);
  for(int rank=0;rank<13;rank++){
    for(int suit=0;suit<default_deck_size/13;suit++){
      deck_.push_back(Card(rank, suit));
    }
  }
  suit_classes_.clear();
  suit_classes_.assign(1, {0,1,2,3}); //at the start, all suits are equivalent
  private_hole_.assign(2, kInvalidHole);
  public_cards_.assign(5, kInvalidCard);

  //AllRolloutEquity(300, 300);

  //AllFlopEquity(200,60,20);
  
  ComputeBuckets(2, 200);

  //TurnHS({Card(6, 0), Card(8, 5), Card(11, 3)}, 1, 1);
  return 0;
}
