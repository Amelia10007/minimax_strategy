mod cow_ref;
mod node;

use cow_ref::CowRef;
use data_structure::Range;
use node::TreeNode;
use num::{Bounded, Integer};
use std::marker::PhantomData;

/// 2人ゲームにおけるプレイヤー．
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Actor {
    /// 先手
    First,
    /// 後手
    Second,
}

impl Actor {
    /// この手番に対する相手側の手番を返す．
    pub fn opponent(&self) -> Self {
        match self {
            Actor::First => Actor::Second,
            Actor::Second => Actor::First,
        }
    }
}

/// ゲームの状態．
pub trait State {}

/// ゲームにおける行動．
pub trait Action {
    /// この行動の手番．
    fn actor(&self) -> Actor;
}

/// ゲーム内の状態遷移条件を記述する．
pub trait Rule {
    /// このゲームルールが考慮するゲームの状態．
    type S;
    /// このゲームルールにおけるプレイヤーの行動．
    type A;
    /// ある状態において実行可能な行動を列挙する際に使用する型．
    type ActionIterator: IntoIterator<Item = Self::A>;

    /// 指定した状態がすでにゲーム終了条件を満たしているか．
    fn is_game_over(state: &Self::S) -> bool;

    /// 指定された状態下で実行可能な行動を列挙する．
    fn iterate_available_actions(state: &Self::S, actor: Actor) -> Self::ActionIterator;

    /// 状態を遷移させる．
    fn translate_state(state: &Self::S, action: &Self::A) -> Self::S;
}

/// ゲーム状態の評価関数．
pub trait Evaluator<S> {
    /// プレイヤーの利得を表す型．
    type Payoff;

    /// 指定された状態について，利得を評価する．
    fn evaluate_payoff_for(actor: Actor, state: &S) -> Self::Payoff;
}

/// ゲームの戦略生成器．
pub trait Strategy<S, A> {
    /// 指定した状態における，指定したエージェントの行動`a`を選択して`Some(a)`として返す．
    /// 取れる行動がない場合は`None`を返す．
    fn select_action(&self, state: &S, actor: Actor) -> Option<A>;
}

/// 2人零和ゲームにおける適切な行動をαβ法で思考するエージェント．
pub struct AlphaBetaStrategy<R, E, N> {
    /// 探索するゲーム木の深さ．
    search_depth: N,
    _r: PhantomData<R>,
    _e: PhantomData<E>,
}

impl<S, A, R, E, N> AlphaBetaStrategy<R, E, N>
where
    S: State,
    A: Action,
    R: Rule<S = S, A = A>,
    E: Evaluator<S>,
    E::Payoff: Copy + Ord + Bounded,
    N: Copy + Integer,
{
    /// αβ法により，指定したノードの評価値を再帰的に計算する．
    /// # Params
    /// 1. remaining_depth 残りの探索深さ．
    /// 1. current_node 注目ノード．
    /// 1. alpha 評価値の関心範囲の下限．
    /// 1. beta 評価値の関心範囲の上限．
    ///
    /// # Returns
    /// `Some(e)`: このノードの評価値`e`
    ///
    /// `None`: このノードがゲーム終了ノードではなく，かつ取れる行動がない場合
    fn construct_best_game_tree_alpha_beta(
        &self,
        remaining_depth: N,
        consideration_target: Actor,
        current_node: &mut TreeNode<MinimaxNode<S, A, E::Payoff>>,
        payoff_range: Range<E::Payoff>,
    ) -> Option<E::Payoff> {
        // デバッグ用アサーション (消しても問題ないけど，コード変更した際の挙動検証のために一応とっておく)
        debug_assert!(current_node.payoff.is_none());

        // 注目ノードが末端ノードなら，現在の状態に対する静的評価値をそのまま適用する
        if remaining_depth.is_zero() || R::is_game_over(current_node.ref_state()) {
            let payoff = E::evaluate_payoff_for(consideration_target, current_node.ref_state());
            current_node.payoff = Some(payoff);
            return Some(payoff);
        }

        // who WILL act on the current state?
        let next_actor = match current_node.cause_action.as_ref() {
            Some(action) => action.actor().opponent(),
            None => consideration_target,
        };

        // 状態遷移などに使用するので，注目ノードの状態をとっておく．
        // ここでは構造体の，後の処理で変更されないメンバだけの参照を保持するだけなので，
        // unsafeブロックの処理は安全である．
        let current_state = {
            let pointer: *const _ = current_node.ref_state();
            unsafe { pointer.as_ref().unwrap() }
        };
        let mut current_payoff_range = payoff_range;

        // 次の実現しうる状態をすべて列挙し，ひとつひとつ調べる
        for mut child in R::iterate_available_actions(current_state, next_actor)
            .into_iter()
            .map(|action| {
                let next_state = R::translate_state(current_state, &action);
                MinimaxNode::new(next_state.into(), Some(action), None)
            })
            .map(|minimax_node| TreeNode::new(minimax_node))
        {
            // 子ノードの評価値を再帰的に求める．
            // ここでNoneが帰ってきた場合，その子ノードはゲーム終了でもなく，かつ取れる行動がないパターンなので，探索対象としない．
            let child_payoff = match self.construct_best_game_tree_alpha_beta(
                remaining_depth - N::one(),
                consideration_target,
                &mut child,
                current_payoff_range,
            ) {
                Some(e) => e,
                None => continue,
            };
            // ミニマックス法により，探索する必要がある枝だけを選択する
            if let Some(e) = current_node.payoff {
                if next_actor == consideration_target {
                    // 自分の手番では，自分が有利になる行動を選択するので，
                    // 自分が不利になる行動は候補から除外する
                    if e >= child_payoff {
                        continue;
                    }
                } else {
                    // 相手の手番では，相手は自分が不利になる行動を選択するので，
                    // 自分が有利になる行動は候補から除外する．
                    if e <= child_payoff {
                        continue;
                    }
                }
            }
            // ここに来たということは，より良い子ノードが見つかったということなので，子ノードの情報を入れ替える．
            // また，注目ノードの評価値には，子ノードの値を反映させる．
            current_node.replace_child(child);
            current_node.payoff = Some(child_payoff);
            // 評価値の注目範囲を更新する．
            // 可能なら，αβカットして探索量を減らす．
            let maybe_next_range = if next_actor == consideration_target {
                Range::try_new(child_payoff, current_payoff_range.max)
            } else {
                Range::try_new(current_payoff_range.min, child_payoff)
            };
            match maybe_next_range {
                Some(range) => current_payoff_range = range,
                None => break,
            }
        }

        // 注目ノードの最終的な評価値を返す．
        // ここに到達した時点で評価値が決定していないということは，
        // 注目ノードの状態から取れる行動がないということなので，
        // そのようなノードは探索の対象にしない．
        current_node.payoff
    }
}

impl<S, A, R, E, N> Strategy<S, A> for AlphaBetaStrategy<R, E, N>
where
    S: State,
    A: Action,
    R: Rule<S = S, A = A>,
    E: Evaluator<S>,
    E::Payoff: Copy + Ord + Bounded,
    N: Copy + Integer,
{
    fn select_action(&self, state: &S, actor: Actor) -> Option<A> {
        let mut root = TreeNode::new(MinimaxNode::<S, A, E::Payoff>::new(
            state.into(),
            None,
            None,
        ));
        self.construct_best_game_tree_alpha_beta(
            self.search_depth,
            actor,
            &mut root,
            Range::new(E::Payoff::min_value(), E::Payoff::max_value()),
        )
        .and_then(|_| root.into_child())
        .and_then(|best_node| best_node.into_inner().cause_action)
    }
}

/// 2人ゲームにおける各プレイヤーを返す．
pub fn actors() -> [Actor; 2] {
    [Actor::First, Actor::Second]
}

pub fn construct_alpha_beta_strategy<R, E, N>(search_depth: N) -> AlphaBetaStrategy<R, E, N> {
    AlphaBetaStrategy {
        search_depth,
        _r: PhantomData,
        _e: PhantomData,
    }
}

/// ミニマックス法で利用するゲーム木のノード．
struct MinimaxNode<'a, S, A, E> {
    /// 現在の状態．
    state: CowRef<'a, S>,
    /// この状態に至る際に実行された行動．
    cause_action: Option<A>,
    /// エージェントにとっての現在状態の評価値．
    payoff: Option<E>,
}

impl<'a, S, A, E> MinimaxNode<'a, S, A, E> {
    fn new(state: CowRef<'a, S>, cause_action: Option<A>, payoff: Option<E>) -> Self {
        Self {
            state,
            cause_action,
            payoff,
        }
    }

    fn ref_state(&self) -> &S {
        self.state.as_ref()
    }
}

#[cfg(test)]
mod test_cmp {
    #[test]
    fn test_cmp() {
        let v = vec![5, 2, 0, 9, 4, 3];
        let max = v.into_iter().max_by(|left, right| left.cmp(right));
        assert_eq!(Some(9), max);
    }
}
