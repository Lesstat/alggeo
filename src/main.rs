#![allow(dead_code)]

use std::marker::PhantomData;

fn main() {
    println!("Hello, world!");
}

pub trait Tree<'items, I>
where
    Self: 'items,
{
    type Query;

    fn build(items: &'items [I], dim: usize) -> Self;
    fn query(&self, query: &Self::Query) -> Vec<I>;
}

pub trait Item: Clone {
    fn index(&self, i: usize) -> usize;
    fn max_dim() -> usize;
}

struct RTreeNode<'a, T, S = Element<'a, T>>
where
    T: Item,
    S: Tree<'a, T>,
{
    left: Option<Box<Self>>,
    max_left: usize,
    sub_tree: S,
    right: Option<Box<Self>>,
    _t: PhantomData<&'a T>,
}

pub struct Element<'a, T>(&'a [T]);

impl<'a, I: Clone> Tree<'a, I> for Element<'a, I> {
    type Query = ();
    fn build(items: &'a [I], _dim: usize) -> Self {
        Element(items)
    }

    fn query(&self, _query: &Self::Query) -> Vec<I> {
        self.0.to_vec()
    }
}

pub struct RTree<'a, I, S = Element<'a, I>>
where
    I: Item,
    S: Tree<'a, I>,
{
    head: RTreeNode<'a, I, S>,
}

impl<'a, I, S> RTree<'a, I, S>
where
    I: Item,
    S: Tree<'a, I>,
{
    fn build(items: &[I]) -> Self {
        todo!()
    }
}

// RTreeNode<Point, TreapNode>::new(&[])
//
struct RTreeQuery<O: Ord> {
    min: O,
    max: O,
}

impl<'a, I, S> RTreeNode<'a, I, S>
where
    I: Item,
    S: Tree<'a, I>,
{
    fn query_left(&self, min: usize, inner_query: &S::Query) -> Vec<I> {
        let mut result = vec![];
        if min <= self.max_left {
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
            result.extend(
                self.right
                    .as_ref()
                    .map(|r| r.query_left(min, inner_query))
                    .unwrap_or_default(),
            );
        }
        result
    }
    fn query_right(&self, max: usize, inner_query: &S::Query) -> Vec<I> {
        let mut result = vec![];
        if self.max_left < max {
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
            result.extend(
                self.left
                    .as_ref()
                    .map(|l| l.query_right(max, inner_query))
                    .unwrap_or_default(),
            );
        }
        result
    }
}

impl<'a, I, S> Tree<'a, I> for RTreeNode<'a, I, S>
where
    I: Item,
    S: Tree<'a, I>,
{
    type Query = (RTreeQuery<usize>, S::Query);
    fn build(items: &'a [I], dim: usize) -> Self {
        let sub_tree = S::build(items, dim + 1);
        if items.len() == 1 {

            // TODO: no sub tree recursion
        }
        let mut items = items.to_vec();
        items.sort_by_key(|i| i.index(dim));

        let median_index = items.len() / 2;
        let median = &items[median_index];
        let max_left = median.index(dim);

        let left = Box::new(Self::build(&items[..=median_index], dim));
        let rigth = Box::new(Self::build(&items[median_index + 1..], dim));

        todo!()
    }

    fn query(&self, query: &Self::Query) -> Vec<I> {
        let my_query = &query.0;

        if my_query.max < self.max_left {
            return self
                .left
                .as_ref()
                .map(|l| l.query(query))
                .unwrap_or_default();
        }

        if self.max_left < my_query.min {
            return self
                .right
                .as_ref()
                .map(|r| r.query(query))
                .unwrap_or_default();
        }

        let mut left_trees = self.query_left(my_query.min, &query.1);
        let right_trees = self.query_right(my_query.max, &query.1);
        left_trees.extend(right_trees.into_iter());
        left_trees
    }
}

#[derive(Clone)]
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

    let tree = RTree::<usize>::build(&items);
    let query = RTreeQuery { min: 2, max: 6 };
    let result = tree.head.query(&(query, ()));

    assert_eq!(vec![3, 5], result);
}
