#![allow(dead_code)]

use std::{collections::HashSet, fmt::Debug, iter::Copied, marker::PhantomData, time::Instant};

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
    fn build_internal(items: &[&'items I], dim: usize) -> Self {
        let mut items: Vec<&'items I> = items.to_vec();
        items.sort_by_key(|i| i.index(dim));
        let head = RTreeNode::build(&items, dim);
        RTree { head }
    }

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
        let (RTreeQuery { min, max }, inner_query) = &self.query;
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
                        self.inner_iter = Some(left
                            .right
                            .as_ref()
                            .map(|r| r.sub_tree.iter_query(inner_query))
                            .unwrap_or_else(|| left.sub_tree.iter_query(inner_query)));
                        self.left = left.left();
                    }
                }

                (None, None, Some(right)) => {
                    if *max <= right.max_left {
                        //println!("expanding right side to the left in iter");
                        self.right = right.left();
                    } else {
                        //println!("expanding right side to the right in iter");
                        self.inner_iter = Some(right
                            .left
                            .as_ref()
                            .map(|r| r.sub_tree.iter_query(inner_query))
                            .unwrap_or_else(|| right.sub_tree.iter_query(inner_query)));
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
    type Query = (RTreeQuery<usize>, S::Query);
    type QueryIter = RTreeIter<'items, 'me, I, S>;

    fn build(items: &[&'items I], dim: usize) -> Self {
        Self::build_internal(items, dim)
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

#[derive(Clone)]
pub struct RTreeQuery<O: Ord> {
    min: O,
    max: O,
}

enum SplitResult<'items, 'me, I: Item, S: Tree<'items, 'me, I>> {
    Split {
        left: &'me RTreeNode<'items, 'me, I, S>,
        right: &'me RTreeNode<'items, 'me, I, S>,
    },
    Final(S::QueryIter),
}

impl<'items, 'me, I, S> RTreeNode<'items, 'me, I, S>
where
    'items: 'me,
    I: Item,
    S: Tree<'items, 'me, I> + Debug,
{
    // type Query = (RTreeQuery<usize>, S::Query);

    fn query_left(&self, min: usize, inner_query: &S::Query) -> Vec<&'items I> {
        // println!("entering query left of: {:?}", self);
        let mut result = vec![];
        if min <= self.max_left {
            // println!("min <= max_left: {} <= {}", min, self.max_left);
            result.extend(
                self.right
                    .as_ref()
                    .map(|r| r.sub_tree.query(inner_query))
                    .unwrap_or_else(|| self.sub_tree.query(inner_query)),
            );
            result.extend(
                self.left
                    .as_ref()
                    .map(|l| l.query_left(min, inner_query))
                    .unwrap_or_default(),
            );
        } else {
            // println!("min > max_left: {} > {}", min, self.max_left);
            result.extend(
                self.right
                    .as_ref()
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
                self.left
                    .as_ref()
                    .map(|l| l.sub_tree.query(inner_query))
                    .unwrap_or_else(|| self.sub_tree.query(inner_query)),
            );
            result.extend(
                self.right
                    .as_ref()
                    .map(|r| r.query_right(max, inner_query))
                    .unwrap_or_default(),
            );
        } else {
            // println!("max_left => max: {} < {}", self.max_left, max);
            result.extend(
                self.left
                    .as_ref()
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

    fn split(
        &'me self,
        query: &<RTree<'items, 'me, I, S> as Tree<'items, 'me, I>>::Query,
    ) -> Option<SplitResult<'items, 'me, I, S>> {
        // println!("entering query at: {:?}", self);
        let my_query = &query.0;

        // println!(
        //     "my_query.max <= self.max_left: {} <= {}",
        //     my_query.max, self.max_left
        // );
        if my_query.max <= self.max_left {
            return self.left.as_ref().and_then(|l| l.split(query));
        }

        // println!(
        //     "self.max_left < my_query.min: {} < {}",
        //     self.max_left, my_query.min
        // );
        if self.max_left < my_query.min {
            return self.right.as_ref().and_then(|r| r.split(query));
        }

        if let (Some(left), Some(right)) = (self.left.as_ref(), self.right.as_ref()) {
            Some(SplitResult::Split { left, right })
        } else {
            Some(SplitResult::Final(self.sub_tree.iter_query(&query.1)))
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
            return self
                .left
                .as_ref()
                .map(|l| l.query(query))
                .unwrap_or_default();
        }

        // println!(
        //     "self.max_left < my_query.min: {} < {}",
        //     self.max_left, my_query.min
        // );
        if self.max_left < my_query.min {
            return self
                .right
                .as_ref()
                .map(|r| r.query(query))
                .unwrap_or_default();
        }

        if let (Some(left), Some(right)) = (self.left.as_ref(), self.right.as_ref()) {
            let mut left_trees = left.query_left(my_query.min, &query.1);
            let right_trees = right.query_right(my_query.max, &query.1);
            left_trees.extend(right_trees.into_iter());
            left_trees
        } else {
            self.sub_tree.query(&query.1)
        }
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
    let query = RTreeQuery { min: 2, max: 6 };
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
        RTreeQuery { min: 2, max: 3 },
        (RTreeQuery { min: 1, max: 2 }, ()),
    );

    let result = tree.query(&query);

    assert_eq!(vec![&items[2]], result);
}

#[test]
fn test_1d_range_tree_iter() {
    let items = [1usize, 3, 5, 7, 8, 9];

    let tree: RTree<usize, Element<usize>> = RTree::build(&items);
    let query = RTreeQuery { min: 2, max: 6 };
    let mut result: Vec<_> = tree.iter_query(&(query, ())).collect();

    result.sort();

    assert_eq!(vec![&items[1], &items[2]], result);
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
        RTreeQuery { min: 2, max: 3 },
        (RTreeQuery { min: 1, max: 2 }, ()),
    );

    let result: Vec<_> = tree.iter_query(&query).collect();

    assert_eq!(vec![&items[2]], result);
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

    let mut min = between.sample(&mut rng);
    let mut max = between.sample(&mut rng);
    if max < min {
        std::mem::swap(&mut min, &mut max);
    }
    let x_query = RTreeQuery { min, max };

    let mut min = between.sample(&mut rng);
    let mut max = between.sample(&mut rng);
    if max < min {
        std::mem::swap(&mut min, &mut max);
    }
    let y_query = RTreeQuery { min, max };

    println!("precalculating expected results");
    let start = Instant::now();
    let expected: Vec<_> = points
        .iter()
        .filter(|&p| {
            x_query.min <= p.x && p.x < x_query.max && y_query.min <= p.y && p.y < y_query.max
        })
        .collect();
    let end = Instant::now();
    let brute_time =  (end - start).as_secs_f64();
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
    println!(
        "querying tree for {} elements took {}s",
        result.len(),
        (end - start).as_secs_f64()
    );

    assert_eq!(expected, result.into_iter().collect());

    println!("querying tree with iterator");
    let start = Instant::now();
    let query = (x_query, (y_query, ()));
    let result = tree.iter_query(&query).collect::<Vec<_>>();
    let end = Instant::now();
    let query_time =  (end - start).as_secs_f64();
    println!(
        "querying tree with iterator for {} elements took {}s",
        result.len(),
        query_time,
    );

    let result_len = result.len();
    assert_eq!(expected, result.into_iter().collect());

    (brute_time, query_time, result_len)
}

fn main() {
    let res: Vec<_> = (10..=19).map(|i| (1 << i, perf_test(1 << i))).collect();
    for (i, (a, b, len)) in res {
      println!("{:08} ({:08}) | {:0.10} {:0.10} {:0.10}", i, len, a, b, b/a)
    }
}
