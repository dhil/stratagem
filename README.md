(*********************************************************************)
(*                                                                   *)
(*                      STRATAGEM Version 1.1                        *)
(*              User documentation and demo examples                 *)
(*                                                                   *)
(*     (C) John Longley, University of Edinburgh, December 2001      *)
(*                      Version 1.1: June 2006                       *)
(*                                                                   *)
(*********************************************************************)
(*                                                                   *)
(* Walkthrough introduction to the STRATAGEM system, featuring some  *)
(* simple ML programs as examples. Tested on NJ-SML v. 110.0.7.      *)
(*                                                                   *)
(*********************************************************************)

STRATAGEM is a research prototype system illustrating some recent
theoretical ideas in game semantics. Given an already compiled ML
function or operation supplied by the user, STRATAGEM extracts the
underlying "computation strategy" and allows it to be explored and
manipulated in various ways. Possible applications include execution
tracing, interactive execution, and program optimization.

The system makes essential use of explicit continuations, and only
runs on the New Jersey SML compiler. (I have tested it on version
110.0.7 - please let me know if you have problems running it on other
versions.)

The STRATAGEM system may be of interest to workers in the following areas:

   * Game semantics. The system provides a "workbench" for one version
     of game semantics, offers a convenient way of exploring the game
     strategy for ML programs, and may be useful as a teaching tool.
   * Program logics and verification. STRATAGEM offers a relatively
     programmer-friendly way of understanding some ideas from semantics
     that are likely to have applications to program logics in the near
     future.
   * Continuations. The system may be of interest as a non-trivial
     example of a program making essential use of continuations.
     It also seems to offer a helpful way of understanding much of what
     continuations do, without the need for explicit continuation types.
   * Execution tracing and debugging tools. STRATAGEM provides a kind
     of execution tracing facility for ML programs, with some unusual
     capabilities (e.g. interactive exploration of alternative execution
     paths) which may be useful for debugging ML programs. Another
     unusual feature is that the execution tracer runs within the same
     ML session as the user's program - perhaps there is some interesting
     way of exploiting this fact?
   * Program optimization. STRATAGEM supplies a "generic optimizer" for
     computation strategies. In some cases, this can lead to a genuine
     speed-up for the user's already compiled program.
   * Lazy and higher order programming. For programs involving lazy data
     structures and higher order functions, it can often be difficult to
     understand their run-time behaviour operationally - "what happens
     when". The execution tracing facilities of STRATAGEM may be useful
     for understanding such programs better.
   * Teaching ML. For newcomers to ML, STRATAGEM might be useful as a
     teaching tool - for example, for viewing the recursion unfoldings of
     a recursively defined function.

The current version (1.1) is intended as a concept demonstrator rather than
a serious debugging tool - the full range of ML datatypes is not yet
supported - but it suffices to illustrate all the basic ideas and allows
the user to perform some complex and interesting experiments.

Getting started
---------------

To run STRATAGEM, you need the New Jersey implementation of ML, and
the source file "stratagem.sml" which is available online at

       http://homepages.inf.ed.ac.uk/jrl/Stratagem/stratagem-1.1.sml

Start up a New Jersey SML session from a directory containing the file
"stratagem.sml". [On the Informatics machines in Edinburgh, this is
done by typing "sml" to the Linux prompt.] Then type:

       use "stratagem-1.1.sml" ;

This builds the system.

You will see that a structure called `Code_Gen` has been opened,
containing (among other things) a recursive datatype `Type`, for
representing the syntax trees of (a class of) ML types.  Values of
type "Type" can be expressed in a form which mimics the ML syntax of
types. For example,

```sml
  (Int Times Bool Arrow Unit) Arrow Int `List ;
```

represents the ML type `(int * bool -> int) -> int list`.

[NOTE: The precedence and binding rules are just as in ML syntax,
 except that only binary products are supported as primitive, so that

```sml
  Int Times Int Times Int
```
corresponds to `(int * int) * int`, rather than to `int * int * int`.
Note also that has been declared as an infix, allowing us to write
Int List instead of List Int.]

Usually one wants to use STRATAGEM in conjunction with an ML program
of some type t determined by the user. To do this, one first needs
to create a bunch of operations specific to the type t.
As an example, suppose we wish to use STRATAGEM with a user program
of type `(int -> int) -> int`. We first perform the following:

```sml
  generate_tools "foo" ((Int Arrow Int) Arrow Int) ;
```

This automatically generates a short source file "foo.sml" (overwriting
any existing file of that name), and loads it in. (If you want to change
the directory into which this file is written, use "chDir" first.)
The file "foo.sml" declares a structure "foo" containing operations
specific to the required type. To make use of this, first type:

```sml
  open foo ;
```
Note that not all ML datatypes are yet supported in this version - not
even all those representable using `Type`. (In fact, the only types
supported at present are equality types and types built up from them
using `->`.) The system will let you know if you try to run
`generate_tools` on a type not yet supported. [There is no obstacle in
principle to supporting all the (functional) datatypes of ML - I just
haven't got round to it yet!]

Execution tracing
-----------------

You may now write any ML program you like of type `(int -> int) -> int`.
As a simple example, consider

```sml
  fun G f = if f 3 < 6 then f 1 else f (f 2 + 1) ;
```

When `G` is applied to a function `f : int -> int`, the resulting computation
can be viewed as a kind of interaction or dialogue between G and f.
The operation foo.trace allows us to see this dialogue unfolding, e.g.

```sml
  trace G (fn x => x+2) ;

  trace G (fn x => x*x) ;
```

What happens here is that "trace `G`" evaluates to an operation whose
behaviour is identical to that of `G`, except that it outputs a lot of
trace information showing the interactions between (the strategy for)
`G` and the rest of the program context.

The form in which the dialogue is displayed may call for some
explanation.  First, the type in question (here, the type of `G`) is
displayed, and under each occurrence of "`->`" we have a (numbered)
column in which moves can occur. The idea is that for a function of
any type `t -> t'`, there are basically two kinds of interaction that
may occur. Either the function may be interrogated for its value at a
certain argument, or the function may return such a value: we think of
these moves as questions (`Q`) of type `t`, or as answers (`A`) of type
`t'`. Depending on the types, both questions or answers can be printable
values of ground types (such as int), or unprintable values of
function types displayed simply as `-`. In the latter case, t' will
itself have the form `u -> u'`, and so more information about the value
represented by `-` can be given by questions and answers in some other
column.

[ NOTE FOR SPECIALISTS: The above conventions are slightly different
  from the usual ones in most papers on game semantics, in which (a)
  the columns are headed by occurrences of ground types rather than
  occurrences of `->`; and (b) questions are not allowed to carry
  values; there is just a single question "?" in each column. These
  differences reflect the way we have chosen to model ML's
  CALL-BY-VALUE evaluation mechanism: one cannot ask a question to a
  function without having a value to offer (operationally, either a
  ground-type value or a closure, i.e. a weak head normal form).  Our
  approach to modelling call-by-value seems close to that of Abramsky
  and McCusker [see Proc. 11th CSL, 1998]. An alternative approach to
  game semantics for call-by-value languages, superficially less close
  to ours, has been proposed by Honda and Yoshida (see Proc. 25th
  ICALP, 1997). ]

Note that moves alternate between two participants: a player `P`,
representing `G` itself, and an opponent `O`, representing the environment
to `G`. In this case, `O` represents the function supplied as an argument
to `G`, as well as the top-level request for the value of `G` when applied
to this function.

Finally, the rightmost column displays what are known as JUSTIFICATION
POINTERS. For every move except the first, the corresponding pointer
indicates the earlier move to which the present move "refers". In the
case of answer moves, the pointer indicates the question (in the same
column) which the move answers. In the case of question moves, the
pointer indicates the move which introduces the function (represented
by `-`) to which the question is being posed. In the case of the
initial question, there is no justification pointer - the function to
which the question is posed is the function we are tracing (`G` in the
above example).

Remarks on execution tracing
----------------------------

1. For the above examples (and those to follow), you should satisfy yourself
   that the trace displayed corresponds to the evaluation behaviour you
   would intuitively expect from your ML program. We hope that in this way
   readers will be able to absorb many of the essential ideas of game
   semantics fairly easily.

2. We hope that our execution tracing facility might (eventually) be
   useful for debugging programs, understanding their operational
   behaviour, and detecting sources of inefficiency.

   With respect to the last of these, however, there is a subtle
   limitation that we should point out. If an ML compiler is clever,
   it might (for all we know) be performing optimizations that take
   advantage of the fact that the user's program is free from side effects,
   e.g. it may eliminate some redundant function calls. However, if we
   now demand a trace for this program, the mere fact of requiring the
   system to output the trace information may be enough to block some of
   these optimizations! Thus, the trace we see when using "trace G" might
   not be a reliable indication of the evaluation behaviour of the pure
   version of `G` after compiler optimization. (In other words, the attempt
   to observe the behaviour of a program may itself affect the thing we
   are trying to observe!)

   [I do not know whether the above possibility can actually arise in the
   case of the New Jersey compiler.]

Interactive mode
----------------

The above shows how we can do execution tracing in "batch mode": we
trace the entire evaluation of an ML expression. However, one can also
step through the execution of a program in interactive mode: here the
machine plays the function we are tracing (such as G), and the user
plays its environment. In terms of the game, the machine is Player and
the user is Opponent. To see this for the above program, type

```sml
  start_game G ;
```

To play the first move, we have to ask the initial question, demanding
the final result of G. We do this by typing

```sml
  Q1 () ;
```

Here the `Q` means we are playing a question; the "1" means we are
playing it in column 1; the `()` is there as a dummy to indicate that
the question carries no printable value (it appears in the trace as
`Q-`).  The system responds by pretty-printing the trace of your move
(labelled as `O1`) together with the machine's response (labelled as
`P1`).  In the case of the above `G`, the machine is posing the
question `Q 3` to the function `-` which we have notionally supplied.

To play the second move, we now need to pretend we are some argument
`f : int -> int`, whose behaviour we may make up as we go along.  For
instance, let us pretend we are the function `(fn x => x+2)`.  We
therefore want to supply the answer `5` to the machine's question.  To
do this we type

```sml
  A0 5 ;
```

which means: play an answer in column 0, carrying the value 5.  Once
again, the system displays our move together with Player's response.

Suppose now that we now change our mind about our last move, or decide
we would like to explore what would have happened if we had played
differently. We can withdraw our last move by typing

```sml
  backup 1 ;
```

which means "backup by one step". The last-but-one pair of moves,
namely `O1` and `P1`, will now be re-displayed, reminding us of the game
state we are now in. Let us now pretend we are the function `(fn x =>
x*x)`. We can now play the alternative move

```sml
  A0 9 ;
```

EXERCISE: continue the game as if you were the playing the function
`(fn x => x*x)`. It will take you two more moves to conclude the game.
You will see that the accumulated trace is the same as that obtained
in batch mode by typing

```sml
  trace G (fn x => x*x) ;
```

Explicit justifiers
-------------------

You will notice that in the above example it was not necessary to
supply explicit justification pointers - the system was able to
compute them automatically. Most of the time this works fine, since
usually there is only one possible justifier for the given move in any
case.  In more complicated cases, however, there is a choice of
possible justifiers, so it is sometimes necessary to supply the
pointers explicitly.  (An example will be given below.) For this
purpose, the system provides primed variants `A0'`,`Q1'` of the
move-playing operations `A0`,`Q1`, which take as an extra argument a
justification pointer - either (init) as a null pointer for the
initial move, or (by `n`) as a pointer to the move `Pn`.  (Note that
Opponent moves, except the initial move, are always justified by
Player moves.)

Thus, the above game could be replayed as follows if we insisted on
supplying the pointers ourselves:

```sml
  start_game G ;
  Q1' () (init) ;
  A0' 9  (by 1) ;
```

etc.

If a "bad" pointer is supplied, the system will usually complain with
an error report.  [NOTE: at present the error trapping on pointers is
not quite complete.  Indeed, the system may sometimes appear to accept
an invalid pointer but will "correct" it to a valid one. We hope these
minor problems will be fixed in the next version!]

Remarks on interactive mode
---------------------------

1. In the current version, only one interactive game play may
   be in progress at a time. An interactive game play is initiated by
   `start_game`; thereafter, the system maintains a stack of game states to
   enable "backup" to work. This stack is cleared the next time `start_game`
   is called.

2. In the above example, the operations `A0` and `Q1` (and their primed variants)
   are declared in the structure `foo`. When the file `foo.sml` is generated,
   the system calculates the columns, types and `Q`/`A` attributes of all possible
   Opponent moves. If the user attempts to play a move which for some reason
   is illegal in the context, the system will (usually) respond with an
   error report.

3. If in an interactive play the system asks us the same question twice,
   there is no reason why we have to supply the same answer both times.
   In other words, we might be pretending we are some argument with mutable
   internal state, whose behaviour can depend on this state. See also the
   remarks on program optimization and the example `H` below.

4. For a similar reason, if the program we are tracing makes use of state,
   its behaviour may depend on the current contents of the state.
   We therefore cannot expect the "backup" facility always to work properly
   for such programs, since the state may have changed since we were last
   in a certain position in the game!

5. It is worth reflecting on what the system is doing during interactive
   mode. In the above example, at each stage the system performs a bit of
   the computation of `G` applied to an argument. Whenever `G` requires
   information about this argument, the state of the whole computation is
   "frozen", and the system jumps out of it and returns to the top-level
   ML prompt. The user can then supply the required piece of information,
   at which point the frozen computation is resumed using this new
   information. It should be clear that explicit continuations are absolutely
   essential to achieve this effect of freezing computations and later
   resuming them.

Format controls
---------------

The structure `Display.Format` contains a few parameters which may be
adjusted to vary the format of the displayed game trace (in both batch
and interactive mode). For instance, to suppress the line of
whitespace between consecutive pairs of moves, type

```sml
  open Display.Format ;
  blank_lines := false ;
```

The parameter `Display.Format.column_width` can be used to vary the
spacing between columns. This might be useful if the values involved
were not just integers but e.g. bool lists, in which case increasing
the column width would stop the columns from running into each other
and improve legibility. (If you require wide columns *and* lots of
them, stretch your window horizontally!)

At the beginning of a (batch or interactive) trace, the system
displays the type in question, together with the associated column
numbers, according to the current format settings. To re-display this
type at any stage, one may call the function "foo.show_type" :

```sml
  show_type () ;
```

(This is useful, for instance, during a long interactive game play
 when the type and column numbers have scrolled off the top of the
 screen.)

Program optimization
--------------------

Another feature of STRATAGEM is that it allows the user's programs to
be automatically "optimized" (in a certain sense) after they have been
compiled. (This application was suggested to us by Mike Fourman.)  For
example, consider the program

```sml
  fun H f = f 3 + f (f 3) ;
```

When `H` is applied to some function `f : int -> int`, the call `f 3` will
normally be performed twice. We can see this by typing e.g.

```sml
  trace H (fn x => x*2) ;
```

Suppose, however, we define

```sml
  val H' = optimize H ;
```

Then `H'` has the same functional behaviour as `H`, but the underlying
strategy has been "optimized" to eliminate redundancies. To see this,
do

```sml
  trace H' (fn x => x*2) ;
```

Note that the second call to `f 3` has been omitted, and the
computation proceeds as if it yielded the same value as the first call
to `f 3`.  The effect is even more marked if we perform

```sml
  trace H  (fn x => x*2-3) ;
  trace H' (fn x => x*2-3) ;
```

Here, three calls to `f 3` are collapsed to a single call. This shows
that the optimization is in some sense happening dynamically - in this
example, it is reminiscent of memoization. Below we will see examples
involving more complex types - in these cases one can perhaps think of
"optimize" as performing some kind of very generic "blanket
memoization" at all possible levels.

Remarks on optimization
-----------------------

1. Note that this kind of optimization will not in general work if there is
   internal state around. If f is an "object" with some internal state that
   may be updated when f is called, there is no guarantee that the value of
   `f 3` for the second time of asking will be the same as for the first,
   so `H f` and `H' f` may yield different results.

   The general point is that if we assume certain constraints on the kind of
   computation involved, this allows us to perform certain kinds of
   optimization. It seems that there are other instances of this phenomenon
   awaiting exploring - what we have done is only a first step in this
   direction.

2. The reader is warned that the above is only "optimizing" `H` in the limited
   sense of minimizing calls to `f`. The process of computing `H'` from `H` is
   quite complex and involves a significant overhead. It will not, in the
   above example, result in a genuine speed-up, although of course it would
   give a genuine improvement in performance in cases where calls to `f` were
   sufficiently costly to compute.

3. Our optimization is not exactly memoization in the usual sense, since
   (in the above example) our H' does not actually involve any persistent
   internal state. Thus, for instance, if we perform
```sml
        trace H' (fn x => x*2) ;
```
   twice, we still need to ask for `f 3` and `f 6` the second time.
   See below for an example of execution tracing for a genuine memoization
   operation.

More examples
-------------

We now present some slightly more complex examples, which serve to
illustrate a few additional points. Just work your way through these
until you get bored ;>)

1. An example involving a slightly more complicated type.
   Perform the following:
```sml
       val ty1 = Int Arrow Int ;
       generate_tools "bar" (ty1 Arrow ty1) ;
       open bar ;
```
The following program illustrates Ulam's famous `3n+1` problem.  The
function "chase" takes a function `f : int -> int` and a starting
value; it iterates `f` until the value `1` is reached, and returns the
number of iterations required. The function "ulam" performs a single
step of the iteration involved in the problem: if the number is even,
halve it, otherwise treble it and add `1`.
```sml
        local
	    fun chase' k f n =
	        if n = 1 then k
	        else chase' (k+1) f (f n)
        in
	    val chase = chase' 0
        end ;

       fun ulam n =
       if n mod 2 = 0
       then n div 2 else n*3+1 ;
```

A famous open problem in mathematics is whether `chase ulam n`
terminates for all integers `n>=1`.

To get an execution trace for this program, type e.g.

```sml
       trace chase ;
       it ulam ;
       it 5 ;
```
This illustrates an important point about the behaviour of (curried)
functions of several arguments in ML. In principle, a bit of
computation happens each time an argument is supplied - it is not all
delayed until a result of ground type is required. (After all, it is
possible in ML to write a function `F : (int->int)->(int->int)` such
that `F ulam` diverges!)

For a more spectacular example, try

       trace chase ulam 27 ;

EXERCISE: Play through the computation of `chase ulam 5` in
interactive mode.

2. A simple example involving ground types other than `int`:
   an instance of the map operation for lists.
```sml
        generate_tools "map"
           ((Int Arrow Bool) Arrow Int `List Arrow Bool `List) ;
        open map ;

        Display.Format.column_width := 20 ;
        trace List.map (fn x => x<=5) [3,8,5] ;
```
EXERCISE: Play through this game in interactive mode.

3. Execution tracing for the standard memoization operation.
   Shows the different call behaviour the second time we ask for some value.
   Uses the same type as example 1 above.
```sml
        val ty1 = Int Arrow Int ;
        generate_tools "bar" (ty1 Arrow ty1) ;
        open bar ;

        (* Usual implementation of memoization for functions int -> int *)
        local
	    datatype 'a option = None | Some of 'a
	    fun lookup x [] = None
	     | lookup x ((y,z)::rest) =
	       if x=y then Some z else lookup x rest
        in
	    fun memo f =
	       let val cache = ref [] in
		   fn x =>
		   (case lookup x (!cache) of
			Some z => z
		      | None =>
			    case f x of z =>
				(cache := (x,z)::(!cache) ; z))
	       end
        end ;

        Display.Format.column_width := 10 ;
        val sq = trace memo (fn x => x * x) ;
        sq 5 ;
        sq 5 ;
```
4. A more complex example: tracing the recursive call structure for the
   (slow) Fibonacci function.
```sml
        val ty1 = Int Arrow Int ;
        generate_tools "zog" ((ty1 Arrow ty1) Arrow ty1) ;
        open zog ;
```
To watch the unfoldings of a recursively defined function of type `int
-> int`, we cannot quite use the already compiled version of the
function, since what we wish to observe are the interactions between
the fixed point operator and the "body" of the function definition.
In this case, we therefore need to edit our source code a little.  In
the case of the Fibonacci function, instead of
```sml
       fun f 0 = 0
         | f 1 = 1
         | f n = f(n-2) + f(n-1) ;
```
we need to declare
```sml
       fun Y F x = F (Y F) x ;
       fun F f 0 = 0
	 | F f 1 = 1
	 | F f n = f(n-2) + f(n-1) ;
```
Clearly `Y F n` yields the nth Fibonacci number. Now try:
```sml
       trace Y F 2 ;
```
Satisfy yourself that the displayed trace is what it should be, and
shows the recursive unwindings of F. If you like, play through the ten
moves of the game in interactive mode.

The length of the game for `Y F n` grows exponentially with `n`.  For
instance, try:
```sml
       trace Y F 10 ;
```
However, here we can use our optimizer to good effect. Try:
```sml
       val Y' = optimize Y ;
       trace Y' F 10 ;
```
This brings the length of the game down to something linear in `n`.  [Of
course, the total run-time of `Y' F n` is still exponential in `n` - it
is only the number of interactions between `Y'` and `F` that is linear!
Even so, this example shows that our optimizer is achieving something;
perhaps the main interest lies in the fact that such a "generic"
optimizer as ours can achieve these results.]

5. An example showing how the use of exceptions and references can give
   rise to games with interesting features.
```sml
        generate_tools "sr" (((Unit Arrow Unit) Arrow Unit) Arrow Bool) ;
        open sr ;
```
Let top, `bot : unit -> unit` be the functions defined by
```sml
       fun top () = () and bot () = bot () ;
```
Consider the function `F` of the above type, specified by
```sml
    (*  F g = true   if g top = (), but g bot diverges;
        F g = false  if g bot = ();
        F g diverges otherwise.                         *)
```
[This is my favourite simple example of a SEQUENTIALLY REALIZABLE
 function. For more information, see the ML source file: "When is a
 functional program not a functional program: a walkthrough
 introduction to the sequentially realizable functionals" available
 from my home page.]

This function may be implemented in ML in various ways.  First, using
a local reference:
```sml
       fun F1 g =
	   let val r = ref false in
	       (g (fn () => r:=true) ; !r)
	   end ;

       trace F1 (fn x => ()) ;
       trace F1 (fn x => x()) ;
```
We can see here that Player's final move is sensitive to what has gone
on between the question and answer in column 1. In game semantics
terms, this is an example of *non-innocent* behaviour - something
generally associated with the presence of state in computations.

Play through the above computations interactively if you like.  Rather
surprisingly perhaps, "backup" works nicely here and seems to restore
the previous contents of the local register r:
```sml
       start_game F1 ;
       Q2 () ;
       Q0 () ;
       backup 1 ;
       A1 () ;
```
(This is something I didn't know about the semantics of continuations.
 It certainly doesn't work like this if you use a global register!)

Now another implementation of F using exceptions.
```sml
       fun F2 g =
	   let exception e in
	       (g (fn () => raise e) ; false)
	       handle e => (g (fn () => ()) ; true)
           end ;
```
Although `F2` computes the same function as `F1`, the underlying computation
strategy is different, as can be seen by trying
```sml
       trace F2 (fn x => x()) ;
```
This example is of interest because the questions `P1` and `O2` are left
"dangling" and never answered. In fact, this is an example of a
*non-well-bracketed* computation, since the final move `P4` does NOT
match the most recent unanswered question. This is a characteristic
generally associated with programs involving control features, such as
exceptions and continuations.

[For the record, there's also an implementation of `F` using
 continuations; its underlying strategy is the same as that of `F2`.]

6. An example in which explicit justification pointers are necessary.
   One needs to go up to a fourth-order ML function for this to happen
   (though morally, the following example lives at call-by-name type 3).
```sml
        generate_tools "four"
          ((((Unit Arrow Unit) Arrow Unit) Arrow Unit) Arrow Unit) ;
        open four ;

        fun Phi F = F (fn t => t()) ;

        fun F1 g = g (fn () => g (fn () => ())) ;
        fun F2 g = g (fn () =>
		     let exception e in
			 (g (fn ()=>raise e)) handle e => ()
		     end) ;
```
Now try
```sml
        trace Phi F1 ;
        trace Phi F2 ;
```
Note that the two plays are identical up to move `O4`, and this move
differs only in its choice of justification pointer.  Suppose we try:
```sml
       start_game Phi ;
```
and play the game interactively until we reach this point.  We then
have a genuine choice whether to play `A0' ()` (by 3) or `A0' ()` (by
2); the system will accept either of these and play the strategy for
`Phi` accordingly.  Note that if we simply enter `A0 ()`, the system
will by default select the MOST RECENT of the possible justifying
moves.  This will usually be the one we want if we are playing a
PCF-style strategy (i.e. one corresponding to a purely functional ML
program).

Even within the world of PCF-style computations, there are cases when
a genuine choice of pointer is necessary, but one needs to go to a
fifth-order ML function to see this!

Rules of the game
-----------------

So far we have not been very precise about the rules applying to games
in our setting - e.g., what constitutes a "legal move" in an
interactive play - but have been relying mainly on ML programmer
intuition.  For advanced experiments, however, one needs to understand
the rules more precisely. For the record, here are the rules for
Opponent moves (i.e. the user's moves), in a rather terse form.

Opponent moves may be of three kinds:
   * The initial move. This must be a question, and must appear in the
     column corresponding to the top-level occurrence of `->` in the
     type in question. The justification pointer must be null, i.e. "init".
   * Non-initial questions. These must appear in a column corresponding
     to a non-top-level occurrence `x` of `->`, and must be justified by a
     player question in the column corresponding to the occurrence of `->`
     directly above `x` in the syntax tree of the type.
   * Answers. These can appear in any column, and must be justified by a
     player question in the same column.

Moreover, the justifying move for a non-initial move must be VISIBLE
at the current point in the game. That is, the justifying move for the
move O(n+1) (where n>=1) must appear in the *O-view* of the sequence
`O1`,`P1`,...,`On`,`Pn`. The O-view of a justified sequence of moves may be
defined inductively as follows:

        O-view () = ()
        O-view (O1,P1,...,Om)    = O-view (O1,...,P(m-1)) . Om
        O-view (O1,P1,...,Om,Pm) = O-view (O1,...,P(k-1)) . Ok . Pm,
                                   where Ok is the justifier of Pm.

Our game protocol therefore coincides with the most liberal of the
four protocols studied in Abramsky and McCusker, "Game semantics"
(Proc. 1995 Marktoberdorf Workshop); see also McCusker's thesis.

How it works (very brief note for specialists!)
-----------------------------------------------

The theory underpinning our system is closely concerned with the
corresponding category in the Abramsky/McCusker framework. The key
observation is that in this category (or at least a closely related
one), there is a universal type, namely
```sml
       datatype forest = forest of nat -> (nat * forest)
```
We can think of this as the type of strategies for the "generic game",
in which two participants take turns to play elements of nat.  The
core of our system is the module `Forest`, which implements the
operations that make the type `forest` into a reflexive object.  Once
this is done, it is not hard to represent all other types t as
computable retracts of `forest`; the embedding half of the retraction
will then be an operation which extracts the underlying computation
strategy of a given operation.

There's a lot more to say here - the full story will be told in detail
in a forthcoming research paper!

Final tips
----------

1. Two common mistakes (in my experience!) which can lead to very bizarre
   results are:
   * Forgetting that you haven't opened an automatically generated structure
     such as `bar`. This can mean that e.g. when playing an interactive
     game, you are inadvertently using operations belonging to a different
     structure, such as `foo`. The results can be most puzzling!
   * Forgetting that you haven't actually initiated an interactive game play,
     Once again, you may find yourself working on a state left over from
     some previous game play.

2. It is possible (though rare in practice) that if you stop the execution
   of some operation using a keyboard interrupt (usually, because it appears
   to be diverging), the system will be left in an inconsistent state,
   which may manifest itself in bizarre outcomes to future experiments.
   If you are worried that this may have happened, type
```sml
        reset () ;
```
   to restore the system to a consistent state.

3. If you are unscrupulous, and don't have much use for the arithmetical
   minus operation, you may perform the following:
```sml
        nonfix - ;
        val - = () ;
```
   This means that in interactive mode you can play moves with non-printable
   values by typing things like `Q1 - ;` instead of `Q1 () ;`.
   This gives the system a somewhat more user-friendly feel.
