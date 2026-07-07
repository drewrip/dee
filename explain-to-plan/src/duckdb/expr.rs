//! Parses the small, SQL-flavored expression language DuckDB embeds in its
//! `extra_info` fields (e.g. `"(grp <= 4)"`, `"__internal_decompress_integral_bigint(#0, 0)"`,
//! `"count_star()"`, `"test.main.t1.id ASC"`) and lowers it to a DataFusion
//! logical [`Expr`].
//!
//! This is *not* a general SQL parser: it covers the operator/function
//! vocabulary DuckDB's physical-plan `EXPLAIN` output actually emits. Columns
//! are referenced either positionally (`#N`, an index into the input
//! schema) or by name (possibly dotted, e.g. `catalog.schema.table.col`,
//! in which case only the last segment is significant).

use std::sync::Arc;

use arrow::datatypes::{DataType, Schema};
use datafusion::common::{Column, DFSchema, ScalarValue};
use datafusion::logical_expr::{
    BinaryExpr, Case, Expr, Operator, WindowFrame, WindowFrameBound, WindowFrameUnits, expr::Between,
};
use datafusion::prelude::SessionContext;

use super::DuckDBTranslateError;

type Result<T> = std::result::Result<T, DuckDBTranslateError>;

/// A parsed function call: `name(arg0, arg1, ...)`, with an optional
/// `DISTINCT` marker (used for aggregates like `count(DISTINCT #0)`).
pub struct ParsedCall {
    pub name: String,
    pub distinct: bool,
    pub args: Vec<Expr>,
}

/// A parsed `<func>(<args>) OVER (...)` window function call.
pub struct ParsedWindowCall {
    pub func_name: String,
    pub distinct: bool,
    pub args: Vec<Expr>,
    pub partition_by: Vec<Expr>,
    /// Sort key, ascending flag, nulls-first flag -- same shape as
    /// [`parse_order_by`]'s return value.
    pub order_by: Vec<(Expr, bool, bool)>,
    pub window_frame: WindowFrame,
}

/// Parses a DuckDB window function projection, e.g.
/// `"ROW_NUMBER() OVER (PARTITION BY grp ORDER BY id ASC NULLS LAST)"` or
/// `"sum(id) OVER (PARTITION BY grp ORDER BY id ASC NULLS LAST ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)"`.
pub fn parse_window_call(text: &str, schema: &Schema, ctx: &SessionContext) -> Result<ParsedWindowCall> {
    let tokens = tokenize(text)?;
    let mut parser = Parser { tokens, pos: 0, schema, ctx };

    let call = parser.parse_call_raw()?;
    if !parser.eat_keyword("OVER") {
        return Err(DuckDBTranslateError::ExprParse(format!("expected OVER in window function: {text:?}")));
    }
    if !parser.eat_punct("(") {
        return Err(DuckDBTranslateError::ExprParse("expected '(' after OVER".into()));
    }

    let mut partition_by = Vec::new();
    if parser.eat_keyword("PARTITION") {
        if !parser.eat_keyword("BY") {
            return Err(DuckDBTranslateError::ExprParse("expected BY after PARTITION".into()));
        }
        loop {
            partition_by.push(parser.parse_or()?);
            if parser.eat_punct(",") {
                continue;
            }
            break;
        }
    }

    let mut order_by = Vec::new();
    if parser.eat_keyword("ORDER") {
        if !parser.eat_keyword("BY") {
            return Err(DuckDBTranslateError::ExprParse("expected BY after ORDER".into()));
        }
        loop {
            let e = parser.parse_or()?;
            let mut asc = true;
            let mut nulls_first = false;
            if parser.eat_keyword("ASC") {
                asc = true;
            } else if parser.eat_keyword("DESC") {
                asc = false;
                nulls_first = true;
            }
            if parser.eat_keyword("NULLS") {
                if parser.eat_keyword("FIRST") {
                    nulls_first = true;
                } else if parser.eat_keyword("LAST") {
                    nulls_first = false;
                }
            }
            order_by.push((e, asc, nulls_first));
            if parser.eat_punct(",") {
                continue;
            }
            break;
        }
    }

    let window_frame = parser.parse_window_frame(!order_by.is_empty())?;

    if !parser.eat_punct(")") {
        return Err(DuckDBTranslateError::ExprParse("expected ')' closing OVER clause".into()));
    }
    parser.expect_end()?;

    Ok(ParsedWindowCall {
        func_name: call.name,
        distinct: call.distinct,
        args: call.args,
        partition_by,
        order_by,
        window_frame,
    })
}

/// Parses `text` as a DuckDB extra_info expression and lowers it to a
/// DataFusion [`Expr`], resolving column references against `schema`.
pub fn parse_expr(text: &str, schema: &Schema, ctx: &SessionContext) -> Result<Expr> {
    let tokens = tokenize(text)?;
    let mut parser = Parser { tokens, pos: 0, schema, ctx };
    let expr = parser.parse_or()?;
    parser.expect_end()?;
    Ok(expr)
}

/// Parses `text` as a single top-level function call (e.g. `"sum_no_overflow(#1)"`,
/// `"count_star()"`) without lowering it to an `Expr` yet -- the caller
/// (aggregate translation) resolves the function name itself.
pub fn parse_call(text: &str, schema: &Schema, ctx: &SessionContext) -> Result<ParsedCall> {
    let tokens = tokenize(text)?;
    let mut parser = Parser { tokens, pos: 0, schema, ctx };
    let call = parser.parse_call_raw()?;
    parser.expect_end()?;
    Ok(call)
}

/// Parses a DuckDB `"Order By"` entry, e.g. `"test.main.t1.id ASC"` or
/// `"grp DESC NULLS LAST"`, returning the sort expression, ascending flag,
/// and nulls-first flag.
pub fn parse_order_by(text: &str, schema: &Schema, ctx: &SessionContext) -> Result<(Expr, bool, bool)> {
    let tokens = tokenize(text)?;
    let mut parser = Parser { tokens, pos: 0, schema, ctx };
    let expr = parser.parse_or()?;

    let mut asc = true;
    let mut nulls_first = false;
    if parser.eat_keyword("ASC") {
        asc = true;
    } else if parser.eat_keyword("DESC") {
        asc = false;
        nulls_first = true; // DuckDB/Postgres default: DESC -> NULLS FIRST
    }
    if parser.eat_keyword("NULLS") {
        if parser.eat_keyword("FIRST") {
            nulls_first = true;
        } else if parser.eat_keyword("LAST") {
            nulls_first = false;
        }
    }
    parser.expect_end()?;
    Ok((expr, asc, nulls_first))
}

// ---------------------------------------------------------------------------
// Tokenizer
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Ident(String),
    Positional(usize),
    Number(String),
    Str(String),
    Punct(&'static str),
}

fn tokenize(text: &str) -> Result<Vec<Token>> {
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;
    let mut tokens = Vec::new();

    while i < chars.len() {
        let c = chars[i];
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        if c == '#' {
            let start = i + 1;
            let mut j = start;
            while j < chars.len() && chars[j].is_ascii_digit() {
                j += 1;
            }
            if j == start {
                return Err(DuckDBTranslateError::ExprParse(format!(
                    "expected digits after '#' in {text:?}"
                )));
            }
            let n: usize = chars[start..j].iter().collect::<String>().parse().unwrap();
            tokens.push(Token::Positional(n));
            i = j;
            continue;
        }
        if c.is_ascii_digit() || (c == '.' && chars.get(i + 1).is_some_and(|c| c.is_ascii_digit())) {
            let start = i;
            let mut j = i;
            while j < chars.len()
                && (chars[j].is_ascii_digit()
                    || chars[j] == '.'
                    || chars[j] == 'e'
                    || chars[j] == 'E'
                    || ((chars[j] == '+' || chars[j] == '-')
                        && j > start
                        && (chars[j - 1] == 'e' || chars[j - 1] == 'E')))
            {
                j += 1;
            }
            tokens.push(Token::Number(chars[start..j].iter().collect()));
            i = j;
            continue;
        }
        if c == '\'' {
            let mut j = i + 1;
            let mut s = String::new();
            loop {
                if j >= chars.len() {
                    return Err(DuckDBTranslateError::ExprParse(format!(
                        "unterminated string literal in {text:?}"
                    )));
                }
                if chars[j] == '\'' {
                    if chars.get(j + 1) == Some(&'\'') {
                        s.push('\'');
                        j += 2;
                        continue;
                    }
                    j += 1;
                    break;
                }
                s.push(chars[j]);
                j += 1;
            }
            tokens.push(Token::Str(s));
            i = j;
            continue;
        }
        if c.is_alphabetic() || c == '_' {
            let start = i;
            let mut j = i;
            while j < chars.len() && (chars[j].is_alphanumeric() || chars[j] == '_' || chars[j] == '.') {
                j += 1;
            }
            tokens.push(Token::Ident(chars[start..j].iter().collect()));
            i = j;
            continue;
        }
        // multi-char punctuation, longest match first
        let multi = ["<=", ">=", "<>", "!=", "||"]
            .into_iter()
            .find(|punct| text_starts_with(&chars, i, punct));
        if let Some(punct) = multi {
            tokens.push(Token::Punct(punct));
            i += punct.len();
            continue;
        }
        let single = ["(", ")", ",", "+", "-", "*", "/", "%", "=", "<", ">"];
        let mut matched = false;
        for punct in single {
            if chars[i] == punct.chars().next().unwrap() {
                tokens.push(Token::Punct(punct_static(punct)));
                i += 1;
                matched = true;
                break;
            }
        }
        if matched {
            continue;
        }
        return Err(DuckDBTranslateError::ExprParse(format!(
            "unexpected character '{c}' in {text:?}"
        )));
    }

    Ok(tokens)
}

fn text_starts_with(chars: &[char], i: usize, s: &str) -> bool {
    let s_chars: Vec<char> = s.chars().collect();
    if i + s_chars.len() > chars.len() {
        return false;
    }
    chars[i..i + s_chars.len()] == s_chars[..]
}

fn punct_static(s: &str) -> &'static str {
    match s {
        "(" => "(",
        ")" => ")",
        "," => ",",
        "+" => "+",
        "-" => "-",
        "*" => "*",
        "/" => "/",
        "%" => "%",
        "=" => "=",
        "<" => "<",
        ">" => ">",
        _ => unreachable!(),
    }
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

struct Parser<'a> {
    tokens: Vec<Token>,
    pos: usize,
    schema: &'a Schema,
    ctx: &'a SessionContext,
}

impl<'a> Parser<'a> {
    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn advance(&mut self) -> Option<Token> {
        let t = self.tokens.get(self.pos).cloned();
        if t.is_some() {
            self.pos += 1;
        }
        t
    }

    fn expect_end(&self) -> Result<()> {
        if self.pos != self.tokens.len() {
            return Err(DuckDBTranslateError::ExprParse(format!(
                "trailing tokens after expression: {:?}",
                &self.tokens[self.pos..]
            )));
        }
        Ok(())
    }

    fn eat_punct(&mut self, p: &str) -> bool {
        if matches!(self.peek(), Some(Token::Punct(x)) if *x == p) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn eat_keyword(&mut self, kw: &str) -> bool {
        if matches!(self.peek(), Some(Token::Ident(s)) if s.eq_ignore_ascii_case(kw)) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn peek_keyword(&self, kw: &str) -> bool {
        matches!(self.peek(), Some(Token::Ident(s)) if s.eq_ignore_ascii_case(kw))
    }

    // Precedence, low to high: OR < AND < NOT < comparison/IS/IN/BETWEEN < || < +- < */ < unary < primary

    fn parse_or(&mut self) -> Result<Expr> {
        let mut lhs = self.parse_and()?;
        while self.eat_keyword("OR") {
            let rhs = self.parse_and()?;
            lhs = Expr::BinaryExpr(BinaryExpr::new(Box::new(lhs), Operator::Or, Box::new(rhs)));
        }
        Ok(lhs)
    }

    fn parse_and(&mut self) -> Result<Expr> {
        let mut lhs = self.parse_not()?;
        while self.eat_keyword("AND") {
            let rhs = self.parse_not()?;
            lhs = Expr::BinaryExpr(BinaryExpr::new(Box::new(lhs), Operator::And, Box::new(rhs)));
        }
        Ok(lhs)
    }

    fn parse_not(&mut self) -> Result<Expr> {
        if self.eat_keyword("NOT") {
            let inner = self.parse_not()?;
            return Ok(Expr::Not(Box::new(inner)));
        }
        self.parse_comparison()
    }

    fn parse_comparison(&mut self) -> Result<Expr> {
        let lhs = self.parse_concat()?;

        if self.eat_keyword("IS") {
            let negated = self.eat_keyword("NOT");
            if self.eat_keyword("DISTINCT") {
                if !self.eat_keyword("FROM") {
                    return Err(DuckDBTranslateError::ExprParse(
                        "expected FROM after IS [NOT] DISTINCT".into(),
                    ));
                }
                let rhs = self.parse_concat()?;
                let op = if negated {
                    Operator::IsNotDistinctFrom
                } else {
                    Operator::IsDistinctFrom
                };
                return Ok(Expr::BinaryExpr(BinaryExpr::new(Box::new(lhs), op, Box::new(rhs))));
            }
            if self.eat_keyword("NULL") {
                return Ok(if negated { lhs.is_not_null() } else { lhs.is_null() });
            }
            return Err(DuckDBTranslateError::ExprParse("expected NULL or DISTINCT after IS".into()));
        }

        let negated = self.eat_keyword("NOT");
        if self.eat_keyword("BETWEEN") {
            let low = self.parse_concat()?;
            if !self.eat_keyword("AND") {
                return Err(DuckDBTranslateError::ExprParse("expected AND in BETWEEN".into()));
            }
            let high = self.parse_concat()?;
            return Ok(Expr::Between(Between {
                expr: Box::new(lhs),
                negated,
                low: Box::new(low),
                high: Box::new(high),
            }));
        }
        if self.eat_keyword("IN") {
            if !self.eat_punct("(") {
                return Err(DuckDBTranslateError::ExprParse("expected '(' after IN".into()));
            }
            let mut list = Vec::new();
            if !self.eat_punct(")") {
                loop {
                    list.push(self.parse_or()?);
                    if self.eat_punct(",") {
                        continue;
                    }
                    break;
                }
                if !self.eat_punct(")") {
                    return Err(DuckDBTranslateError::ExprParse("expected ')' after IN list".into()));
                }
            }
            return Ok(Expr::InList(datafusion::logical_expr::expr::InList {
                expr: Box::new(lhs),
                list,
                negated,
            }));
        }
        if negated {
            return Err(DuckDBTranslateError::ExprParse(
                "dangling NOT (expected BETWEEN/IN)".into(),
            ));
        }

        let op = match self.peek() {
            Some(Token::Punct("=")) => Some(Operator::Eq),
            Some(Token::Punct("<>")) | Some(Token::Punct("!=")) => Some(Operator::NotEq),
            Some(Token::Punct("<")) => Some(Operator::Lt),
            Some(Token::Punct("<=")) => Some(Operator::LtEq),
            Some(Token::Punct(">")) => Some(Operator::Gt),
            Some(Token::Punct(">=")) => Some(Operator::GtEq),
            _ => None,
        };
        if let Some(op) = op {
            self.pos += 1;
            let rhs = self.parse_concat()?;
            return Ok(Expr::BinaryExpr(BinaryExpr::new(Box::new(lhs), op, Box::new(rhs))));
        }

        Ok(lhs)
    }

    fn parse_concat(&mut self) -> Result<Expr> {
        let mut lhs = self.parse_additive()?;
        while self.eat_punct("||") {
            let rhs = self.parse_additive()?;
            lhs = Expr::BinaryExpr(BinaryExpr::new(
                Box::new(lhs),
                Operator::StringConcat,
                Box::new(rhs),
            ));
        }
        Ok(lhs)
    }

    fn parse_additive(&mut self) -> Result<Expr> {
        let mut lhs = self.parse_multiplicative()?;
        loop {
            let op = match self.peek() {
                Some(Token::Punct("+")) => Operator::Plus,
                Some(Token::Punct("-")) => Operator::Minus,
                _ => break,
            };
            self.pos += 1;
            let rhs = self.parse_multiplicative()?;
            lhs = Expr::BinaryExpr(BinaryExpr::new(Box::new(lhs), op, Box::new(rhs)));
        }
        Ok(lhs)
    }

    fn parse_multiplicative(&mut self) -> Result<Expr> {
        let mut lhs = self.parse_unary()?;
        loop {
            let op = match self.peek() {
                Some(Token::Punct("*")) => Operator::Multiply,
                Some(Token::Punct("/")) => Operator::Divide,
                Some(Token::Punct("%")) => Operator::Modulo,
                _ => break,
            };
            self.pos += 1;
            let rhs = self.parse_unary()?;
            lhs = Expr::BinaryExpr(BinaryExpr::new(Box::new(lhs), op, Box::new(rhs)));
        }
        Ok(lhs)
    }

    fn parse_unary(&mut self) -> Result<Expr> {
        if self.eat_punct("-") {
            let inner = self.parse_unary()?;
            return Ok(Expr::Negative(Box::new(inner)));
        }
        self.parse_primary()
    }

    fn parse_primary(&mut self) -> Result<Expr> {
        match self.advance() {
            Some(Token::Positional(n)) => self.resolve_positional(n),
            Some(Token::Number(n)) => Ok(Expr::Literal(parse_number(&n), None)),
            Some(Token::Str(s)) => Ok(Expr::Literal(ScalarValue::Utf8(Some(s)), None)),
            Some(Token::Punct("(")) => {
                let e = self.parse_or()?;
                if !self.eat_punct(")") {
                    return Err(DuckDBTranslateError::ExprParse("expected ')'".into()));
                }
                Ok(e)
            }
            Some(Token::Ident(name)) => self.parse_ident_expr(name),
            other => Err(DuckDBTranslateError::ExprParse(format!(
                "unexpected token {other:?} while parsing expression"
            ))),
        }
    }

    fn parse_ident_expr(&mut self, name: String) -> Result<Expr> {
        match name.to_ascii_uppercase().as_str() {
            "TRUE" => return Ok(Expr::Literal(ScalarValue::Boolean(Some(true)), None)),
            "FALSE" => return Ok(Expr::Literal(ScalarValue::Boolean(Some(false)), None)),
            "NULL" => return Ok(Expr::Literal(ScalarValue::Null, None)),
            "CAST" => return self.parse_cast(),
            "CASE" => return self.parse_case(),
            _ => {}
        }

        if self.eat_punct("(") {
            let mut distinct = false;
            if self.peek_keyword("DISTINCT") {
                self.pos += 1;
                distinct = true;
            }
            let mut args = Vec::new();
            if !self.eat_punct(")") {
                loop {
                    args.push(self.parse_or()?);
                    if self.eat_punct(",") {
                        continue;
                    }
                    break;
                }
                if !self.eat_punct(")") {
                    return Err(DuckDBTranslateError::ExprParse(format!(
                        "expected ')' closing call to {name}"
                    )));
                }
            }
            return self.resolve_scalar_call(&name, distinct, args);
        }

        self.resolve_named_column(&name)
    }

    fn parse_call_raw(&mut self) -> Result<ParsedCall> {
        let name = match self.advance() {
            Some(Token::Ident(n)) => n,
            other => {
                return Err(DuckDBTranslateError::ExprParse(format!(
                    "expected function name, found {other:?}"
                )));
            }
        };
        if !self.eat_punct("(") {
            return Err(DuckDBTranslateError::ExprParse(format!("expected '(' after {name}")));
        }
        let mut distinct = false;
        if self.peek_keyword("DISTINCT") {
            self.pos += 1;
            distinct = true;
        }
        let mut args = Vec::new();
        if !self.eat_punct(")") {
            loop {
                args.push(self.parse_or()?);
                if self.eat_punct(",") {
                    continue;
                }
                break;
            }
            if !self.eat_punct(")") {
                return Err(DuckDBTranslateError::ExprParse(format!("expected ')' closing call to {name}")));
            }
        }
        Ok(ParsedCall { name, distinct, args })
    }

    /// Parses an optional `ROWS|RANGE|GROUPS [BETWEEN <bound> AND <bound> | <bound>]`
    /// frame clause. If absent, returns the SQL-standard default frame for
    /// whether an `ORDER BY` is present.
    fn parse_window_frame(&mut self, has_order_by: bool) -> Result<WindowFrame> {
        let units = if self.eat_keyword("ROWS") {
            WindowFrameUnits::Rows
        } else if self.eat_keyword("RANGE") {
            WindowFrameUnits::Range
        } else if self.eat_keyword("GROUPS") {
            WindowFrameUnits::Groups
        } else {
            return Ok(WindowFrame::new(Some(has_order_by)));
        };

        if self.eat_keyword("BETWEEN") {
            let start = self.parse_frame_bound()?;
            if !self.eat_keyword("AND") {
                return Err(DuckDBTranslateError::ExprParse("expected AND in frame BETWEEN clause".into()));
            }
            let end = self.parse_frame_bound()?;
            Ok(WindowFrame::new_bounds(units, start, end))
        } else {
            let start = self.parse_frame_bound()?;
            Ok(WindowFrame::new_bounds(units, start, WindowFrameBound::CurrentRow))
        }
    }

    fn parse_frame_bound(&mut self) -> Result<WindowFrameBound> {
        if self.eat_keyword("UNBOUNDED") {
            if self.eat_keyword("PRECEDING") {
                return Ok(WindowFrameBound::Preceding(ScalarValue::UInt64(None)));
            }
            if self.eat_keyword("FOLLOWING") {
                return Ok(WindowFrameBound::Following(ScalarValue::UInt64(None)));
            }
            return Err(DuckDBTranslateError::ExprParse("expected PRECEDING or FOLLOWING after UNBOUNDED".into()));
        }
        if self.eat_keyword("CURRENT") {
            if !self.eat_keyword("ROW") {
                return Err(DuckDBTranslateError::ExprParse("expected ROW after CURRENT".into()));
            }
            return Ok(WindowFrameBound::CurrentRow);
        }
        let n = match self.advance() {
            Some(Token::Number(s)) => s,
            other => {
                return Err(DuckDBTranslateError::ExprParse(format!(
                    "expected a frame bound (UNBOUNDED/CURRENT ROW/<n> PRECEDING|FOLLOWING), found {other:?}"
                )));
            }
        };
        let value: u64 = n.parse().map_err(|_| {
            DuckDBTranslateError::ExprParse(format!("expected an integer frame bound, found '{n}'"))
        })?;
        if self.eat_keyword("PRECEDING") {
            Ok(WindowFrameBound::Preceding(ScalarValue::UInt64(Some(value))))
        } else if self.eat_keyword("FOLLOWING") {
            Ok(WindowFrameBound::Following(ScalarValue::UInt64(Some(value))))
        } else {
            Err(DuckDBTranslateError::ExprParse(format!(
                "expected PRECEDING or FOLLOWING after frame bound '{n}'"
            )))
        }
    }

    fn parse_cast(&mut self) -> Result<Expr> {
        if !self.eat_punct("(") {
            return Err(DuckDBTranslateError::ExprParse("expected '(' after CAST".into()));
        }
        let inner = self.parse_or()?;
        if !self.eat_keyword("AS") {
            return Err(DuckDBTranslateError::ExprParse("expected AS in CAST".into()));
        }
        let type_name = self.parse_type_name()?;
        if !self.eat_punct(")") {
            return Err(DuckDBTranslateError::ExprParse("expected ')' closing CAST".into()));
        }
        let dt = map_type_name(&type_name)?;
        Ok(Expr::Cast(datafusion::logical_expr::Cast::new(Box::new(inner), dt)))
    }

    fn parse_type_name(&mut self) -> Result<String> {
        match self.advance() {
            Some(Token::Ident(n)) => {
                let mut full = n;
                if self.eat_punct("(") {
                    // e.g. DECIMAL(18,3) -- consume and ignore precision/scale
                    let mut depth = 1;
                    while depth > 0 {
                        match self.advance() {
                            Some(Token::Punct("(")) => depth += 1,
                            Some(Token::Punct(")")) => depth -= 1,
                            Some(_) => {}
                            None => {
                                return Err(DuckDBTranslateError::ExprParse(
                                    "unterminated type parameters".into(),
                                ));
                            }
                        }
                    }
                    full.push_str("()");
                }
                Ok(full)
            }
            other => Err(DuckDBTranslateError::ExprParse(format!("expected type name, found {other:?}"))),
        }
    }

    fn parse_case(&mut self) -> Result<Expr> {
        let mut operand = None;
        if !self.peek_keyword("WHEN") {
            operand = Some(Box::new(self.parse_or()?));
        }
        let mut when_then: Vec<(Box<Expr>, Box<Expr>)> = Vec::new();
        while self.eat_keyword("WHEN") {
            let cond = self.parse_or()?;
            if !self.eat_keyword("THEN") {
                return Err(DuckDBTranslateError::ExprParse("expected THEN in CASE".into()));
            }
            let val = self.parse_or()?;
            when_then.push((Box::new(cond), Box::new(val)));
        }
        let else_expr = if self.eat_keyword("ELSE") {
            Some(Box::new(self.parse_or()?))
        } else {
            None
        };
        if !self.eat_keyword("END") {
            return Err(DuckDBTranslateError::ExprParse("expected END closing CASE".into()));
        }
        Ok(Expr::Case(Case {
            expr: operand,
            when_then_expr: when_then,
            else_expr,
        }))
    }

    fn resolve_positional(&self, n: usize) -> Result<Expr> {
        let field = self.schema.fields().get(n).ok_or_else(|| {
            DuckDBTranslateError::ExprParse(format!(
                "positional reference #{n} out of range for schema with {} columns",
                self.schema.fields().len()
            ))
        })?;
        Ok(Expr::Column(Column::new_unqualified(field.name())))
    }

    fn resolve_named_column(&self, name: &str) -> Result<Expr> {
        let leaf = name.rsplit('.').next().unwrap_or(name);
        if self.schema.fields().iter().any(|f| f.name() == leaf) {
            return Ok(Expr::Column(Column::new_unqualified(leaf)));
        }
        Err(DuckDBTranslateError::ExprParse(format!(
            "column '{name}' not found in input schema {:?}",
            self.schema.fields().iter().map(|f| f.name()).collect::<Vec<_>>()
        )))
    }

    fn resolve_scalar_call(&self, name: &str, _distinct: bool, args: Vec<Expr>) -> Result<Expr> {
        if is_internal_wrapper(name) {
            return args.into_iter().next().ok_or_else(|| {
                DuckDBTranslateError::ExprParse(format!("{name} expects at least one argument"))
            });
        }

        let mapped = map_scalar_fn_name(name);
        use datafusion::execution::FunctionRegistry;
        let udf = self.ctx.udf(mapped).map_err(|_| {
            DuckDBTranslateError::UnsupportedFunction(name.to_string())
        })?;
        Ok(Expr::ScalarFunction(datafusion::logical_expr::expr::ScalarFunction { func: udf, args }))
    }
}

fn parse_number(raw: &str) -> ScalarValue {
    if raw.contains('.') || raw.to_ascii_lowercase().contains('e') {
        ScalarValue::Float64(raw.parse().ok())
    } else if let Ok(i) = raw.parse::<i64>() {
        ScalarValue::Int64(Some(i))
    } else {
        ScalarValue::Float64(raw.parse().ok())
    }
}

/// DuckDB emits synthetic `__internal_*(expr, ...)` calls when reading from
/// compressed/dictionary/bitpacked storage — e.g.
/// `__internal_decompress_integral_bigint(#0, 0)`,
/// `__internal_compress_string_utinyint(#0, ...)`. These describe *storage*
/// encoding decisions the query never mentions and have no corresponding
/// relational operator or SQL expression; whatever type suffix follows
/// `__internal_{de,}compress_` (integral, string, or any future variant) is
/// storage metadata, not a value transformation. They lower to a pure
/// passthrough of their first argument — the actual column/expression being
/// encoded — dropping every other argument (offsets, dictionaries, etc.).
fn is_internal_wrapper(name: &str) -> bool {
    name.starts_with("__internal")
}

fn map_scalar_fn_name(name: &str) -> &str {
    match name.to_ascii_lowercase().as_str() {
        "substring" => "substr",
        other => Box::leak(other.to_string().into_boxed_str()),
    }
}

fn map_type_name(name: &str) -> Result<DataType> {
    let base = name.trim_end_matches("()");
    Ok(match base.to_ascii_uppercase().as_str() {
        "TINYINT" | "INT1" => DataType::Int8,
        "SMALLINT" | "INT2" => DataType::Int16,
        "INTEGER" | "INT" | "INT4" => DataType::Int32,
        "BIGINT" | "INT8" => DataType::Int64,
        "UTINYINT" => DataType::UInt8,
        "USMALLINT" => DataType::UInt16,
        "UINTEGER" => DataType::UInt32,
        "UBIGINT" => DataType::UInt64,
        "HUGEINT" => DataType::Decimal128(38, 0),
        "FLOAT" | "REAL" => DataType::Float32,
        "DOUBLE" => DataType::Float64,
        "VARCHAR" | "TEXT" | "STRING" | "CHAR" | "BPCHAR" => DataType::Utf8,
        "BOOLEAN" | "BOOL" => DataType::Boolean,
        "DATE" => DataType::Date32,
        "TIMESTAMP" => DataType::Timestamp(arrow::datatypes::TimeUnit::Microsecond, None),
        "DECIMAL" | "NUMERIC" => DataType::Decimal128(38, 10),
        other => {
            return Err(DuckDBTranslateError::ExprParse(format!("unsupported cast target type: {other}")));
        }
    })
}

/// Converts a lowered logical [`Expr`] into a `DFSchema`-bound physical
/// expression against `schema`.
pub fn to_physical(
    expr: &Expr,
    schema: &Schema,
) -> Result<Arc<dyn datafusion::physical_plan::PhysicalExpr>> {
    let df_schema = DFSchema::try_from(schema.clone())
        .map_err(|e| DuckDBTranslateError::ExprParse(e.to_string()))?;
    datafusion::physical_expr::create_physical_expr(
        expr,
        &df_schema,
        &datafusion::logical_expr::execution_props::ExecutionProps::new(),
    )
    .map_err(|e| DuckDBTranslateError::ExprParse(e.to_string()))
}
