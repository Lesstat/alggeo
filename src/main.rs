#![allow(dead_code)]

use std::{
    cmp::Ordering, collections::HashSet, fmt::Debug, iter::Copied, marker::PhantomData,
    time::Instant,
};

use rand::{
    distributions::Uniform,
    prelude::{Distribution, StdRng},
    thread_rng, Rng, SeedableRng,
};
use smallvec::SmallVec;

pub trait Tree<'items, 'me, I: 'items>
where
    Self: 'me,
{
    type Query: Clone;
    type QueryIter: Iterator<Item = &'items I>;

    fn build(items: &[&'items I], dim: usize) -> Self;
    fn query(&self, query: &Self::Query) -> Vec<&'items I>;
    fn iter_query(&'me self, query: &Self::Query) -> Self::QueryIter;
}

pub trait Item: Debug {
    fn index(&self, i: usize) -> usize;
    fn max_dim() -> usize;
}

struct RTreeNode<'items, 'me, T, S = Element<'items, T>>
where
    T: Item,
    S: Tree<'items, 'me, T>,
{
    left: Option<Box<Self>>,
    max_left: usize,
    sub_tree: S,
    right: Option<Box<Self>>,
    _t: PhantomData<(&'me T, &'items T)>,
}

impl<'items, 'me, T, S> RTreeNode<'items, 'me, T, S>
where
    T: Item,
    S: Tree<'items, 'me, T>,
{
    fn left(&self) -> Option<&Self> {
        self.left.as_ref().map(|b| b.as_ref())
    }

    fn right(&self) -> Option<&Self> {
        self.right.as_ref().map(|b| b.as_ref())
    }
}

#[derive(Debug)]
struct FCHints {
    left: usize,
    right: usize,
}

#[derive(Debug)]
struct FCRTreeNode<'items, 'me, T>
where
    T: Item,
{
    left: Option<Box<Self>>,
    max_left: usize,
    elements: Vec<&'items T>,
    hints: Vec<FCHints>, // maybe len is elements.len + 1
    // query.min, query.max
    // start = elements.binary_search(query.min)
    // end = elements.binary_search(query.max)
    // yield return elemnts[start..end]
    // ...
    // left.query(start.left, end.left)
    // ...
    right: Option<Box<Self>>,
    _t: PhantomData<(&'me T, &'items T)>,
}

#[derive(Debug)]
pub struct Element<'items, I>(SmallVec<[&'items I; 2]>);

impl<'items, 'me, I: Debug> Tree<'items, 'me, I> for Element<'items, I>
where
    'items: 'me,
{
    type Query = ();
    type QueryIter = Copied<std::slice::Iter<'me, &'items I>>;

    fn build(items: &[&'items I], _dim: usize) -> Self {
        Element(items.into())
    }

    fn query(&self, _query: &Self::Query) -> Vec<&'items I> {
        self.0.to_vec()
    }
    fn iter_query(&'me self, _: &Self::Query) -> Self::QueryIter {
        self.0.iter().copied()
    }
}

#[derive(Debug)]
pub struct RTree<'items, 'me, I, S = Element<'items, I>>
where
    I: Item,
    S: Tree<'items, 'me, I>,
{
    head: RTreeNode<'items, 'me, I, S>,
}

impl<'items, 'me, I, S> RTree<'items, 'me, I, S>
where
    'items: 'me,
    I: Item,
    S: Tree<'items, 'me, I> + Debug,
{
    fn build(items: &'items [I]) -> Self {
        let items: Vec<_> = items.iter().collect();
        Tree::build(&items, 0)
    }
}

pub struct FCRTree<'items, 'me, I>
where
    I: Item,
{
    head: FCRTreeNode<'items, 'me, I>,
}

impl<'items, 'me, I> FCRTree<'items, 'me, I>
where
    'items: 'me,
    I: Item,
{
    fn build(items: &'items [I]) -> Self {
        let items: Vec<_> = items.iter().collect();
        Tree::build(&items, 0)
    }
}

pub struct RTreeIter<'items, 'me, I: Item, S: Tree<'items, 'me, I> + Debug> {
    left: Option<&'me RTreeNode<'items, 'me, I, S>>,
    right: Option<&'me RTreeNode<'items, 'me, I, S>>,
    inner_iter: Option<S::QueryIter>,
    query: <RTree<'items, 'me, I, S> as Tree<'items, 'me, I>>::Query,
}

impl<'items, 'me, I: Item, S: Tree<'items, 'me, I> + Debug> RTreeIter<'items, 'me, I, S> {
    fn new(
        left: Option<&'me RTreeNode<'items, 'me, I, S>>,
        right: Option<&'me RTreeNode<'items, 'me, I, S>>,
        query: <RTree<'items, 'me, I, S> as Tree<'items, 'me, I>>::Query,
    ) -> Self {
        Self {
            left,
            right,
            inner_iter: None,
            query,
        }
    }
    fn empty(query: <RTree<'items, 'me, I, S> as Tree<'items, 'me, I>>::Query) -> Self {
        //println!("returning empty iter");
        Self {
            left: None,
            right: None,
            inner_iter: None,
            query,
        }
    }
}

impl<'items, 'me, I, S> Iterator for RTreeIter<'items, 'me, I, S>
where
    I: Item,
    S: Tree<'items, 'me, I> + Debug,
{
    type Item = &'items I;

    fn next(&mut self) -> Option<Self::Item> {
        let (RangedQuery { min, max }, inner_query) = &self.query;
        loop {
            //println!(
            //    "iter state: inner: {:?}, left: {:?}, right: {:?}",
            //    self.inner_iter.is_some(),
            //    self.left,
            //    self.right
            //);
            match (self.inner_iter.as_mut(), self.left, self.right) {
                // We still have elements in the inner iter
                (Some(ref mut iter), _, _) => {
                    //println!("return element from inner iter");
                    if let Some(it) = iter.next() {
                        //println!("coming back with {:?}", it);
                        break Some(it);
                    } else {
                        //println!("inner iter is empty");
                        self.inner_iter = None;
                    }
                }
                // No elements in inner iter, try to go left
                (None, Some(left), _) => {
                    if left.max_left < *min {
                        //println!("expanding left side to the right in iter");
                        self.left = left.right();
                    } else {
                        //println!("expanding left side to the left in iter");
                        self.inner_iter = Some(
                            left.right()
                                .map(|r| r.sub_tree.iter_query(inner_query))
                                .unwrap_or_else(|| left.sub_tree.iter_query(inner_query)),
                        );
                        self.left = left.left();
                    }
                }

                (None, None, Some(right)) => {
                    if *max <= right.max_left {
                        //println!("expanding right side to the left in iter");
                        self.right = right.left();
                    } else {
                        //println!("expanding right side to the right in iter");
                        self.inner_iter = Some(
                            right
                                .left()
                                .map(|r| r.sub_tree.iter_query(inner_query))
                                .unwrap_or_else(|| right.sub_tree.iter_query(inner_query)),
                        );
                        self.right = right.right();
                    }
                }
                (None, None, None) => break None,
            }
        }
    }
}

impl<'me, 'items, S, I> Tree<'items, 'me, I> for RTree<'items, 'me, I, S>
where
    'items: 'me,
    I: Item,
    S: Tree<'items, 'me, I> + Debug,
{
    type Query = (RangedQuery<usize>, S::Query);
    type QueryIter = RTreeIter<'items, 'me, I, S>;

    fn build(items: &[&'items I], dim: usize) -> Self {
        let mut items: Vec<&'items I> = items.to_vec();
        items.sort_by_key(|i| i.index(dim));
        let head = RTreeNode::build(&items, dim);
        RTree { head }
    }
    fn query(&self, query: &Self::Query) -> Vec<&'items I> {
        self.head.query(query)
    }
    fn iter_query(&'me self, query: &Self::Query) -> Self::QueryIter {
        let mut cur = &self.head;
        let my_query = &query.0;

        loop {
            if my_query.max <= cur.max_left {
                if let Some(left) = cur.left() {
                    //println!("going down and left");
                    cur = left;
                    continue;
                } else {
                    return RTreeIter::empty(query.clone());
                }
            }

            if cur.max_left < my_query.min {
                if let Some(right) = cur.right() {
                    //println!("going down and right");
                    cur = right;
                    continue;
                } else {
                    return RTreeIter::empty(query.clone());
                }
            }
            //println!("starting iteration at {:?}", cur);
            break;
        }

        if cur.left().is_none() || cur.right().is_none() {
            assert!(cur.left().is_none() && cur.right().is_none());
            RTreeIter::new(Some(cur), None, query.clone())
        } else {
            RTreeIter::new(cur.left(), cur.right(), query.clone())
        }
    }
}

impl<'me, 'items, I> Tree<'items, 'me, I> for FCRTree<'items, 'me, I>
where
    'items: 'me,
    I: Item,
{
    type Query = (RangedQuery<usize>, RangedQuery<usize>);
    type QueryIter = std::iter::Empty<&'items I>;

    fn build(items: &[&'items I], dim: usize) -> Self {
        let mut lower_items: Vec<&'items I> = items.to_vec();
        let mut items: Vec<&'items I> = items.to_vec();
        items.sort_by_key(|i| i.index(dim));
        lower_items.sort_by_key(|i| i.index(dim + 1));
        let head = FCRTreeNode::build(&items, lower_items, dim);
        Self { head }
    }
    fn query(&self, query: &Self::Query) -> Vec<&'items I> {
        self.head.query(query)
    }
    fn iter_query(&'me self, query: &Self::Query) -> Self::QueryIter {
        unimplemented!("{:?}", query.0.min);
    }
}

#[derive(Clone, Debug)]
pub struct RangedQuery<O: Ord> {
    min: O,
    max: O,
}

impl<'items, 'me, I, S> RTreeNode<'items, 'me, I, S>
where
    'items: 'me,
    I: Item,
    S: Tree<'items, 'me, I> + Debug,
{
    // type Query = (RangedQuery<usize>, S::Query);

    fn query_left(&self, min: usize, inner_query: &S::Query) -> Vec<&'items I> {
        // println!("entering query left of: {:?}", self);
        let mut result = vec![];
        if min <= self.max_left {
            // println!("min <= max_left: {} <= {}", min, self.max_left);
            result.extend(
                self.right()
                    .map(|r| r.sub_tree.query(inner_query))
                    .unwrap_or_else(|| self.sub_tree.query(inner_query)),
            );
            result.extend(
                self.left()
                    .map(|l| l.query_left(min, inner_query))
                    .unwrap_or_default(),
            );
        } else {
            // println!("min > max_left: {} > {}", min, self.max_left);
            result.extend(
                self.right()
                    .map(|r| r.query_left(min, inner_query))
                    .unwrap_or_default(),
            );
        }
        result
    }
    fn query_right(&self, max: usize, inner_query: &S::Query) -> Vec<&'items I> {
        // println!("entering query right of: {:?}", self);
        let mut result = vec![];
        if self.max_left < max {
            // println!("max_left < max: {} < {}", self.max_left, max);
            result.extend(
                self.left()
                    .map(|l| l.sub_tree.query(inner_query))
                    .unwrap_or_else(|| self.sub_tree.query(inner_query)),
            );
            result.extend(
                self.right()
                    .map(|r| r.query_right(max, inner_query))
                    .unwrap_or_default(),
            );
        } else {
            // println!("max_left => max: {} < {}", self.max_left, max);
            result.extend(
                self.left()
                    .map(|l| l.query_right(max, inner_query))
                    .unwrap_or_default(),
            );
        }
        result
    }

    fn build(items: &[&'items I], dim: usize) -> Self {
        // println!("building node at dim {} with items: {:?}", dim, items);
        let sub_tree = S::build(items, dim + 1);

        if items.len() == 1 {
            let max_left = items[0].index(dim);
            // println!("max_left: {}", max_left);
            return Self {
                left: None,
                right: None,
                max_left,
                sub_tree,
                _t: PhantomData,
            };
        }

        let median_index = (items.len() - 1) / 2;
        let median = &items[median_index];
        let max_left = median.index(dim);
        // println!("max_left: {}", max_left);

        let left = Box::new(Self::build(&items[..=median_index], dim));
        let right = Box::new(Self::build(&items[median_index + 1..], dim));

        Self {
            left: Some(left),
            right: Some(right),
            max_left,
            sub_tree,
            _t: PhantomData,
        }
    }

    fn query(
        &self,
        query: &<RTree<'items, 'me, I, S> as Tree<'items, 'me, I>>::Query,
    ) -> Vec<&'items I> {
        // println!("entering query at: {:?}", self);
        let my_query = &query.0;

        // println!(
        //     "my_query.max <= self.max_left: {} <= {}",
        //     my_query.max, self.max_left
        // );
        if my_query.max <= self.max_left {
            return self.left().map(|l| l.query(query)).unwrap_or_default();
        }

        // println!(
        //     "self.max_left < my_query.min: {} < {}",
        //     self.max_left, my_query.min
        // );
        if self.max_left < my_query.min {
            return self.right().map(|r| r.query(query)).unwrap_or_default();
        }

        if let (Some(left), Some(right)) = (self.left(), self.right()) {
            let mut left_trees = left.query_left(my_query.min, &query.1);
            let right_trees = right.query_right(my_query.max, &query.1);
            left_trees.extend(right_trees.into_iter());
            left_trees
        } else {
            self.sub_tree.query(&query.1)
        }
    }
}

impl<'items, 'me, I> FCRTreeNode<'items, 'me, I>
where
    'items: 'me,
    I: Item,
{
    fn build(items: &[&'items I], lower_items: Vec<&'items I>, dim: usize) -> Self {
        if items.len() == 1 {
            let max_left = items[0].index(dim);
            // println!("max_left: {}", max_left);
            return Self {
                left: None,
                right: None,
                max_left,
                elements: lower_items,
                hints: vec![],
                _t: PhantomData,
            };
        }

        let median_index = (items.len() - 1) / 2;
        let median = &items[median_index];
        let max_left = median.index(dim);
        // println!("max_left: {}", max_left);

        let mut left_items = Vec::with_capacity(median_index + 1);
        let mut right_items = Vec::with_capacity(items.len() - median_index - 1);
        let mut hints = Vec::with_capacity(items.len() + 1);

        let right_median_points: HashSet<_> = items[median_index + 1..]
            .iter()
            .take_while(|i| i.index(dim) == median.index(dim))
            .map(|&i| i as *const _)
            .collect();

        for &e in &lower_items {
            hints.push(FCHints {
                left: left_items.len(),
                right: right_items.len(),
            });
            match e.index(dim).cmp(&median.index(dim)) {
                Ordering::Less => left_items.push(e),
                Ordering::Equal => {
                    if right_median_points.contains(&(e as *const _)) {
                        right_items.push(e);
                    } else {
                        left_items.push(e);
                    }
                }
                Ordering::Greater => right_items.push(e),
            }
        }
        hints.push(FCHints {
            left: left_items.len(),
            right: right_items.len(),
        });

        let left = Box::new(Self::build(&items[..=median_index], left_items, dim));
        let right = Box::new(Self::build(&items[median_index + 1..], right_items, dim));

        Self {
            left: Some(left),
            right: Some(right),
            max_left,
            elements: lower_items,
            hints,
            _t: PhantomData,
        }
    }

    fn left(&self) -> Option<&Self> {
        self.left.as_ref().map(|b| b.as_ref())
    }

    fn right(&self) -> Option<&Self> {
        self.right.as_ref().map(|b| b.as_ref())
    }

    fn query_left(&self, min: usize, start_hint: usize, end_hint: usize) -> Vec<&'items I> {
        // println!("entering query left of: {:?}", self);
        // println!(
        //     "query_left: start_hint {:?}, end {:?}",
        //     start_hint, end_hint
        // );
        if end_hint - start_hint == 0 {
            return vec![];
        }
        if self.right.is_none() {
            return self.elements[start_hint..end_hint]
                .iter()
                .filter(|e| min <= e.index(I::max_dim() - 2))
                .copied()
                .collect::<Vec<_>>();
        }
        let start = &self.hints[start_hint];
        let end = &self.hints[end_hint];
        let mut result = vec![];
        if min <= self.max_left {
            // println!("min <= max_left: {} <= {}", min, self.max_left);
            result.extend(
                self.right()
                    .map(|r| {
                        r.elements[start.right..end.right]
                            .iter()
                            .copied()
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default(),
            );
            result.extend(
                self.left()
                    .map(|l| l.query_left(min, start.left, end.left))
                    .unwrap_or_default(),
            );
        } else {
            // println!("min > max_left: {} > {}", min, self.max_left);
            result.extend(
                self.right()
                    .map(|l| l.query_left(min, start.right, end.right))
                    .unwrap_or_default(),
            );
        }
        result
    }
    fn query_right(&self, max: usize, start_hint: usize, end_hint: usize) -> Vec<&'items I> {
        // println!("entering query left of: {:?}", self);
        // println!(
        //     "query_right: start_hint {:?}, end {:?}",
        //     start_hint, end_hint
        // );
        if end_hint - start_hint == 0 {
            return vec![];
        }
        if self.right.is_none() {
            return self.elements[start_hint..end_hint]
                .iter()
                .filter(|e| e.index(I::max_dim() - 2) < max)
                .copied()
                .collect::<Vec<_>>();
        }
        let start = &self.hints[start_hint];
        let end = &self.hints[end_hint];
        let mut result = vec![];
        if self.max_left < max {
            // println!("max <= max_left: {} <= {}", max, self.max_left);
            result.extend(
                self.left()
                    .map(|r| r.elements[start.left..end.left].iter().collect::<Vec<_>>())
                    .unwrap_or_default(),
            );
            result.extend(
                self.right()
                    .map(|l| l.query_right(max, start.right, end.right))
                    .unwrap_or_default(),
            );
        } else {
            // println!("max > max_left: {} > {}", max, self.max_left);
            result.extend(
                self.left()
                    .map(|l| l.query_right(max, start.left, end.left))
                    .unwrap_or_default(),
            );
        }
        result
    }

    //    X-Tree              Y-Tree
    //         2              .1.224
    //     1       3      .1.2      ..24
    //   1   2   3   4  :2   .1.  .4   :2

    fn query(
        &self,
        query: &<FCRTree<'items, 'me, I> as Tree<'items, 'me, I>>::Query,
    ) -> Vec<&'items I> {
        // println!("entering query at: {:?}", self);
        let my_query = &query.0;
        let lower_query = &query.1;

        let lower_dim = I::max_dim() - 1;
        let my_dim = lower_dim - 1;

        if self.right.is_none() {
            assert!(self.left.is_none());
            assert_eq!(self.elements.len(), 1);
            return self
                .elements
                .iter()
                .filter(|e| {
                    my_query.min <= e.index(my_dim)
                        && e.index(my_dim) < my_query.max
                        && lower_query.min <= e.index(lower_dim)
                        && e.index(lower_dim) < lower_query.max
                })
                .copied()
                .collect::<Vec<_>>();
        }
        // println!(
        //     "my_query.max <= self.max_left: {} <= {}",
        //     my_query.max, self.max_left
        // );
        if my_query.max <= self.max_left {
            return self.left().map(|l| l.query(query)).unwrap_or_default();
        }

        // println!(
        //     "self.max_left < my_query.min: {} < {}",
        //     self.max_left, my_query.min
        // );
        if self.max_left < my_query.min {
            return self.right().map(|r| r.query(query)).unwrap_or_default();
        }
        let lower_dim = I::max_dim() - 1;

        let start_hint =
            lower_bound_by_key(&self.elements, &lower_query.min, |e| e.index(lower_dim));
        let end_hint = lower_bound_by_key(&self.elements, &lower_query.max, |e| e.index(lower_dim));
        // println!(
        //     "TreeNode::query: start_hint {:?}, end {:?}, my_qurey {:?}, low_q {:?}",
        //     start_hint, end_hint, my_query, lower_query
        // );
        // println!(
        //     "{:?}",
        //     self.elements
        //         .iter()
        //         .map(|e| e.index(lower_dim))
        //         .collect::<Vec<_>>()
        // );
        let start = &self.hints[start_hint];
        let end = &self.hints[end_hint];
        if let (Some(left), Some(right)) = (self.left(), self.right()) {
            let mut left_trees = left.query_left(my_query.min, start.left, end.left);
            let right_trees = right.query_right(my_query.max, start.right, end.right);
            left_trees.extend(right_trees.into_iter());
            left_trees
        } else {
            // self.sub_tree.query(&query.1)
            unimplemented!()
        }
    }
}

fn lower_bound_by_key<'a, E: Debug, B: Ord, F: FnMut(&E) -> B>(
    v: &'a [E],
    b: &B,
    mut key: F,
) -> usize {
    let res = v.binary_search_by_key(b, &mut key);
    // println!("{:?}", res);
    match res {
        Err(idx) => idx,
        Ok(idx) => v[..idx]
            .iter()
            .rev()
            .enumerate()
            // .inspect(|(i, e)| println!("before skip {:?},{:?}", i, e))
            .skip_while(|(_, e)| key(e) == *b)
            // .inspect(|(i, e)| println!("before map {:?},{:?}", i, e))
            .map(|(i, _)| idx - i)
            // .inspect(|i| println!("after map {:?}", i))
            .next()
            .unwrap_or(0),
    }
}

impl<'items, 'me, T: Item, S: Tree<'items, 'me, T> + Debug> Debug for RTreeNode<'items, 'me, T, S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.sub_tree)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Point {
    x: usize,
    y: usize,
}

impl Item for Point {
    fn index(&self, i: usize) -> usize {
        match i {
            0 => self.x,
            1 => self.y,
            _ => panic!("not that big of a dimension"),
        }
    }

    fn max_dim() -> usize {
        2
    }
}

impl Item for usize {
    fn index(&self, i: usize) -> usize {
        assert!(i == 0);
        *self
    }

    fn max_dim() -> usize {
        1
    }
}

#[test]
fn test_1d_range_tree() {
    let items = [1usize, 3, 5, 7, 8, 9];

    let tree: RTree<usize, Element<usize>> = RTree::build(&items);
    let query = RangedQuery { min: 2, max: 6 };
    let mut result = tree.head.query(&(query, ()));

    result.sort();

    assert_eq!(vec![&items[1], &items[2]], result);
}

#[test]
fn test_2d_range_tree() {
    let items = [
        Point { x: 1, y: 2 },
        Point { x: 3, y: 4 },
        Point { x: 2, y: 1 },
        Point { x: 4, y: 2 },
    ];

    let tree: RTree<Point, RTree<_>> = RTree::build(&items);
    let query = (
        RangedQuery { min: 2, max: 3 },
        (RangedQuery { min: 1, max: 2 }, ()),
    );

    let result = tree.query(&query);

    assert_eq!(vec![&items[2]], result);
}

#[test]
fn test_1d_range_tree_iter() {
    let items = [1usize, 3, 5, 7, 8, 9];

    let tree: RTree<usize, Element<usize>> = RTree::build(&items);
    let query = RangedQuery { min: 2, max: 6 };
    let mut result: Vec<_> = tree.iter_query(&(query, ())).collect();

    result.sort();

    assert_eq!(vec![&items[1], &items[2]], result);
}

#[test]
fn test_fc_2d_range_tree() {
    let items = [
        Point { x: 1, y: 2 },
        Point { x: 3, y: 4 },
        Point { x: 2, y: 1 },
        Point { x: 4, y: 2 },
    ];
    //         2            1224
    //     1       3      12   24
    //   1   2   3   4   2  1 4  2
    //
    //         2              1.22.4
    //     1       3      1.2.      .2.4
    //   1   2   3   4  .2.   1:  :4   .2.
    //
    //         2              .1.224
    //     1       3      .1.2      ..24
    //   1   2   3   4  :2   .1.  .4   :2
    //

    let tree: FCRTree<Point> = FCRTree::build(&items);
    let query = (
        RangedQuery { min: 2, max: 3 },
        RangedQuery { min: 1, max: 2 },
    );

    let result: Vec<_> = tree.query(&query);

    assert_eq!(vec![&Point { x: 2, y: 1 }], result);
}

#[test]
fn test_fc_2d_range_tree_xquery_is_respected() {
    let items = [Point { x: 28, y: 49 }, Point { x: 50, y: 43 }];

    let tree: FCRTree<Point> = FCRTree::build(&items);
    let query = (
        RangedQuery { min: 4, max: 42 },
        RangedQuery { min: 39, max: 63 },
    );

    let result: Vec<_> = tree.query(&query);

    assert_eq!(vec![&Point { x: 28, y: 49 }], result);
}

#[test]
fn test_2d_range_tree_iter() {
    let items = [
        Point { x: 1, y: 2 },
        Point { x: 3, y: 4 },
        Point { x: 2, y: 1 },
        Point { x: 4, y: 2 },
    ];

    let tree: RTree<Point, RTree<_>> = RTree::build(&items);
    let query = (
        RangedQuery { min: 2, max: 3 },
        (RangedQuery { min: 1, max: 2 }, ()),
    );

    let result: Vec<_> = tree.iter_query(&query).collect();

    assert_eq!(vec![&items[2]], result);
}

#[test]
#[ignore]
fn test_2d_fctree_leaf_should_have_one_element() {
    let seed = 16975172300249298234;
    let point_count = 1048576;

    let mut rng = StdRng::seed_from_u64(seed);
    let between = Uniform::from(1..100);

    println!("generating points");
    let points: Vec<_> = (0..point_count)
        .map(|_| {
            let x = between.sample(&mut rng);
            let y = between.sample(&mut rng);
            Point { x, y }
        })
        .collect();

    println!("building fc tree");
    let start = Instant::now();
    // let tree: RTree<Point, RTree<_>> = RTree::build(&points);
    let fc_tree = FCRTree::build(&points);
    let end = Instant::now();
    println!("Building tree took {}s", (end - start).as_secs_f64());

    let mut min = between.sample(&mut rng);
    let mut max = between.sample(&mut rng);
    if max < min {
        std::mem::swap(&mut min, &mut max);
    }
    let x_query = RangedQuery { min, max };

    let mut min = between.sample(&mut rng);
    let mut max = between.sample(&mut rng);
    if max < min {
        std::mem::swap(&mut min, &mut max);
    }
    let y_query = RangedQuery { min, max };

    println!("querying fc tree");
    let start = Instant::now();
    let query = (x_query.clone(), (y_query.clone()/*, ()*/));
    let result = fc_tree.query(&query);
    let end = Instant::now();

    let fc_query_time = (end - start).as_secs_f64();
    println!(
        "querying tree for {} elements took {}s",
        result.len(),
        fc_query_time,
    );

    let res_vec: Vec<_> = result.into_iter().collect();
    // if expected != res_vec {
    dbg!(&points);
    dbg!(&query);
    // dbg!(&expected);
    dbg!(&res_vec);
    // }
}

fn perf_test(point_count: usize) -> (f64, f64, usize) {
    let mut thread_rng = thread_rng();
    let seed = thread_rng.gen();
    println!("used seed {} for {} elems", seed, point_count);

    let mut rng = StdRng::seed_from_u64(seed);
    let between = Uniform::from(1..100);

    println!("generating points");
    let points: Vec<_> = (0..point_count)
        .map(|_| {
            let x = between.sample(&mut rng);
            let y = between.sample(&mut rng);
            Point { x, y }
        })
        .collect();

    println!("building tree");
    let start = Instant::now();
    let tree: RTree<Point, RTree<_>> = RTree::build(&points);
    let end = Instant::now();
    println!("Building tree took {}s", (end - start).as_secs_f64());

    println!("building fc tree");
    let start = Instant::now();
    // let tree: RTree<Point, RTree<_>> = RTree::build(&points);
    let fc_tree = FCRTree::build(&points);
    let end = Instant::now();
    println!("Building tree took {}s", (end - start).as_secs_f64());

    let mut min = between.sample(&mut rng);
    let mut max = between.sample(&mut rng);
    if max < min {
        std::mem::swap(&mut min, &mut max);
    }
    let x_query = RangedQuery { min, max };

    let mut min = between.sample(&mut rng);
    let mut max = between.sample(&mut rng);
    if max < min {
        std::mem::swap(&mut min, &mut max);
    }
    let y_query = RangedQuery { min, max };

    println!("precalculating expected results");
    let start = Instant::now();
    let expected: Vec<_> = points
        .iter()
        .filter(|&p| {
            x_query.min <= p.x && p.x < x_query.max && y_query.min <= p.y && p.y < y_query.max
        })
        .collect();
    let end = Instant::now();
    let brute_time = (end - start).as_secs_f64();
    println!(
        "precalc for {} elments took {}s",
        expected.len(),
        brute_time,
    );

    let expected: HashSet<_> = expected.into_iter().collect();

    println!("querying tree");
    let start = Instant::now();
    let query = (x_query.clone(), (y_query.clone(), ()));
    let result = tree.query(&query);
    let end = Instant::now();
    let query_time = (end - start).as_secs_f64();
    println!(
        "querying tree for {} elements took {}s",
        result.len(),
        query_time,
    );

    let res_vec = result.into_iter().collect();
    if expected != res_vec {
        dbg!(&points);
        dbg!(&query);
        dbg!(&expected);
        dbg!(&res_vec);
    }
    println!("querying fc tree");
    let start = Instant::now();
    let query = (x_query.clone(), (y_query.clone()/*, ()*/));
    let result = fc_tree.query(&query);
    let end = Instant::now();

    let fc_query_time = (end - start).as_secs_f64();
    println!(
        "querying tree for {} elements took {}s",
        result.len(),
        fc_query_time,
    );

    let res_vec = result.into_iter().collect();
    if expected != res_vec {
        dbg!(&points);
        dbg!(&query);
        dbg!(&expected);
        dbg!(&res_vec);
    }

    // println!("querying tree with iterator");
    // let start = Instant::now();
    // let query = (x_query, (y_query, ()));
    // let result = tree.iter_query(&query).collect::<Vec<_>>();
    // let end = Instant::now();
    // let query_time = (end - start).as_secs_f64();
    // println!(
    //     "querying tree with iterator for {} elements took {}s",
    //     result.len(),
    //     query_time,
    // );

    // let result_len = result.len();
    // assert_eq!(expected, result.into_iter().collect());

    (query_time, fc_query_time, expected.len())
}

fn main() {
    let res: Vec<_> = (10..=20).map(|i| (1 << i, perf_test(1 << i))).collect();
    for (i, (a, b, len)) in res {
        println!(
            "{:08} ({:08}) | {:0.10} {:0.10} {:0.10}",
            i,
            len,
            a,
            b,
            a / b
        )
    }
}

pub fn is_sorted_by_key<T, F, K>(slice: &[T], f: F) -> bool
where
    F: Fn(&T) -> K,
    K: PartialOrd,
{
    slice.windows(2).all(|w| f(&w[0]) <= f(&w[1]))
}
