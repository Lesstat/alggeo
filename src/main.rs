#![allow(dead_code)]

use std::{collections::HashSet, fmt::Debug, marker::PhantomData, time::Instant};

use rand::{
    distributions::Uniform,
    prelude::{Distribution, StdRng},
    thread_rng, Rng, SeedableRng,
};

pub trait Tree<'items, I>
where
    Self: 'items,
{
    type Query;

    fn build(items: &[&'items I], dim: usize) -> Self;
    fn query(&self, query: &Self::Query) -> Vec<&'items I>;
}

pub trait Item: Debug {
    fn index(&self, i: usize) -> usize;
    fn max_dim() -> usize;
}

struct RTreeNode<'items, T, S = Element<'items, T>>
where
    T: Item,
    S: Tree<'items, T>,
{
    left: Option<Box<Self>>,
    max_left: usize,
    sub_tree: S,
    right: Option<Box<Self>>,
    _t: PhantomData<&'items T>,
}

// TODO how to resolve liftimes within Trees:
//  1) raw point in Element in Tree::items
//  2) Tree::items never move b/c all Trees are not unpin
//  3) that would delete 'tree everywhere
//  4) profit
//  OR simpler
//  try Vec<&'items T> as Element

#[derive(Debug)]
pub struct Element<'items, T>(Vec<&'items T>);

impl<'items, I: Debug> Tree<'items, I> for Element<'items, I> {
    type Query = ();
    fn build(items: &[&'items I], _dim: usize) -> Self {
        Element(items.to_vec())
    }

    fn query(&self, _query: &Self::Query) -> Vec<&'items I> {
        println!("adding {:?} to output", self.0);
        self.0.clone()
    }
}

#[derive(Debug)]
pub struct RTree<'items, I, S = Element<'items, I>>
where
    I: Item,
    S: Tree<'items, I>,
{
    head: RTreeNode<'items, I, S>,
}

impl<'items, I, S> RTree<'items, I, S>
where
    I: Item,
    S: Tree<'items, I> + Debug,
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

impl<'items, I, S> Tree<'items, I> for RTree<'items, I, S>
where
    I: Item,
    S: Tree<'items, I> + Debug,
{
    fn build(items: &[&'items I], dim: usize) -> Self {
        Self::build_internal(items, dim)
    }
    fn query(&self, query: &Self::Query) -> Vec<&'items I> {
        self.head.query(query)
    }

    type Query = (RTreeQuery<usize>, S::Query);
}

// RTreeNode<Point, TreapNode>::new(&[])
//
pub struct RTreeQuery<O: Ord> {
    min: O,
    max: O,
}

impl<'items, I, S> RTreeNode<'items, I, S>
where
    I: Item,
    S: Tree<'items, I> + Debug,
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

    fn query(&self, query: &<RTree<'items, I, S> as Tree<'items, I>>::Query) -> Vec<&'items I> {
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

impl<'items, T: Item, S: Tree<'items, T> + Debug> Debug for RTreeNode<'items, T, S> {
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

fn main() {
    let point_count = 1_000_000;
    let mut thread_rng = thread_rng();
    let seed = thread_rng.gen();
    println!("used seed {}", seed);

    let mut rng = StdRng::seed_from_u64(seed);
    let between = Uniform::from(1..10000);

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

    let x_query = RTreeQuery {
        min: between.sample(&mut rng),
        max: between.sample(&mut rng),
    };
    let y_query = RTreeQuery {
        min: between.sample(&mut rng),
        max: between.sample(&mut rng),
    };

    println!("precalculating expected results");
    let start = Instant::now();
    let expected: HashSet<_> = points
        .iter()
        .filter(|&p| {
            x_query.min <= p.x && p.x < x_query.max && y_query.min <= p.y && p.y < y_query.max
        })
        .collect();
    let end = Instant::now();
    println!("precalc took {}s", (end - start).as_secs_f64());

    println!("querying tree");
    let start = Instant::now();
    let query = (x_query, (y_query, ()));
    let end = Instant::now();
    println!("querying tree took {}s", (end - start).as_secs_f64());

    let result: HashSet<_> = tree.query(&query).into_iter().collect();

    assert_eq!(expected, result);
}
