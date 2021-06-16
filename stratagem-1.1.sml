(*********************************************************************)
(*                                                                   *)
(*                      STRATAGEM Version 1.1                        *)
(*                      Complete source code                         *)
(*                                                                   *)
(*     (C) John Longley, University of Edinburgh, December 2001      *)
(*                      Version 1.1: June 2006                       *)
(*                                                                   *)
(*********************************************************************)
(*                                                                   *)
(* Complete ML source code for the STRATAGEM system, Version 1.0.    *)
(* Contains signature declarations, functor bodies, build commands.  *)
(* Requires New Jersey ML (uses continuations).                      *)
(* Tested on NJ-SML v. 110.0.7.                                      *)
(* Version 1.1: Arguments to "Display" functor modified to allow     *) 
(* compilation on v. 110.42 (thanks to Ethan Aubin for help).        *)
(* See the accompanying file "userdoc.txt" for background and        *)
(* user documentation.                                               *)
(*                                                                   *)
(*********************************************************************)

(* For easy rebuild: *)
fun build () = use "stratagem.sml" ;

(* Turn off garbage collector messages, they interfere with output: *)
val GC_messages = SMLofNJ.Internals.GC.messages ;
GC_messages false ;


(* SIGNATURE DECLARATIONS *)

(* Natural numbers with coding functions.
   Actually, we only require an infinite set X equipped with
   codings for X*X, X+X, X list, and an injection int -> X. *)

infix lt gt leq geq

signature NAT_IO = sig             (* conversion between natural numbers *)
    eqtype nat                     (* and other types *)
    val nat : int -> nat
    val int : nat -> int
    val str : nat -> string
    val ev : string -> nat
    exception Nat
end

signature PAIR = sig               (* bijective coding NxN <-> N *)
    type nat
    val pair : nat -> nat -> nat
    val proj : nat -> (nat * nat)
    val fst : nat -> nat           (* implementations of fst,snd may be *)
    val snd : nat -> nat           (* faster than that of proj *)
end

signature CASES = sig              (* bijective coding N+N <-> N *)
    type nat
    val inl : nat -> nat
    val inr : nat -> nat
    val cases : nat -> (nat * bool)
    val outlr : nat -> nat
    val isl : nat -> bool
end

signature LST = sig                (* bijective coding List[N] <-> N *)
    type nat
    val empty : nat
    val cons : nat -> nat -> nat
    val dest : nat -> (nat * nat)
    val head : nat -> nat
    val tail : nat -> nat
    val isEmpty : nat -> bool
    exception Empty
end

signature NAT = sig
    type nat
    structure Nat_IO : NAT_IO
    structure Pair : PAIR
    structure Cases : CASES
    structure Lst : LST
    sharing type nat = Nat_IO.nat = Pair.nat = Cases.nat = Lst.nat
end

(* Enhanced version including some utilities definable from the above *)

signature NAT_UTILS = sig 
    include NAT
    val n0 : nat 
    val n1 : nat 
    val codeList : nat list -> nat
    val decodeList : nat -> nat list
    val quest : nat -> nat
    val ans : nat -> nat
    val deQuest : nat -> nat
    val deAns : nat -> nat
    val isQuest : nat -> bool
    val isAns : nat -> bool
end


(* Lambda algebra of lazy forests *)

infix % 
signature FOREST = sig
   type nat
   datatype forest = forest of nat -> (nat * forest)
   val %      : forest * nat -> nat * forest
   val apply  : forest -> forest -> forest
   val lambda : (forest -> forest) -> forest
   val reset  : unit -> bool
end 

signature IRRED = sig
   type forest
   val irred  : forest -> forest
end 

(* Stuff for representing general ML types as retracts of forest. *)

signature RETRACTS = sig

   type nat
   type forest
   datatype ('a,'b) sum = left of 'a | right of 'b

   type ('a,'b) retraction = ('a -> 'b) * ('b -> 'a)
   type 'a coding = ('a,nat) retraction
   type 'a rep = ('a,forest) retraction

   val unit_coding : unit coding
   val bool_coding : bool coding
   val int_coding  : int coding
   val nat_coding  : nat coding
   val product_coding : ''a coding -> ''b coding -> (''a * ''b) coding
   val sum_coding     : ''a coding -> ''b coding -> (''a,''b) sum coding
   val list_coding    : ''a coding -> ''a list coding

   val arrow0_rep  : ''a coding -> ''b coding -> (''a -> ''b) rep
   val arrow1_rep  : ''a coding ->  'b rep    -> (''a ->  'b) rep
   val arrow2_rep  :  'a rep ->    ''b coding -> ( 'a -> ''b) rep
   val arrow3_rep  :  'a rep ->     'b rep    -> ( 'a ->  'b) rep

end


(* Stuff for interpreting forest dialogues as game plays for general
   ML types. Closely related to RETRACTS. *)

infix \ $

signature DECODER = sig

   type unraveller
   val arrow0_unraveller : int -> unraveller
   val arrow1_unraveller : int -> unraveller -> unraveller
   val arrow2_unraveller : int -> unraveller -> unraveller
   val arrow3_unraveller : int -> unraveller -> unraveller -> unraveller
(* val triv_unraveller   : int -> unraveller *)

   type nat
   type time
   val timeToInt : time -> int
   datatype QA = Q | A
   val QA_string : QA -> string
   type move_info = QA * nat * int * time * time option

   type printer   = move_info -> string

   type decoder
   val make_decoder : unraveller -> decoder
   val \ : decoder * nat -> move_info * decoder

end


(* The other direction: stuff for translating a game play for the ML type
   into a forest dialogue. *)

signature ENCODER = sig

   type nat
   type encoder 
   type move_info = nat * int * int * int option
   val $ : encoder * move_info -> nat * encoder

   type column_info = int * int * int
   val arrow0_encoder : column_info -> encoder
   val arrow1_encoder : column_info -> encoder -> encoder
   val arrow2_encoder : column_info -> encoder -> encoder
   val arrow3_encoder : column_info -> encoder -> encoder -> encoder

   exception Column
   exception Expired
   exception Bad_Pointer
   exception Wrong_Pointer

end

(* Syntax of ML types supported. Operations involving computation
   over the syntax of types. *)

infix 7 `
infix 5 Times
infixr 3 Arrow
infix \

signature TYPES = sig

   type nat
   datatype Type =
        Unit | Bool | Int | Nat 
      | Arrow of Type * Type
      | Times of Type * Type
      | Sum of Type * Type
      | List of Type
      | Triv (* for debugging *)

   exception Type_Error
   val ` : 'a * ('a -> Type) -> Type
   val ML_for_Type      : Type -> string
   val syntax_for_Type  : Type -> string
   val rep_for_Type     : Type -> string

   type decoder
   type encoder
   type printer
   val decoding   : Type -> decoder * int * printer
   val encoding   : Type -> encoder

   type Type0
   type QA
   val coding_for : Type0 -> string
   val O_types_in : Type -> (QA * int * Type0) list

end


(* Displaying game traces *)

signature DISPLAY = sig

   type nat
   type forest
   type Type
   type 'a rep = ('a -> forest) * (forest -> 'a)

   val display_type      : Type -> unit
   val trace_strategy    : 'a rep -> Type -> 'a -> 'a
   val start_interaction : 'a rep -> Type -> 'a -> unit
   val play_move         : nat * int * int option -> unit
   val play_auto_move    : nat * int -> unit

   structure Interact_Tools : sig
      val backup : int -> unit
      val init   : int option
      val by     : int -> int option
   end

   (* formatting parameters *)
   structure Format : sig
      val column_width   : int ref
      val left_margin    : int ref
      val pointer_margin : int ref
      val blank_lines    : bool ref
   end

end


(* Stuff for generating and loading ML code specific to the 
   type specified by the user *)

signature CODE_GEN = sig
datatype Type =
     Unit | Bool | Int | Nat 
   | Arrow of Type * Type
   | Times of Type * Type
   | Sum of Type * Type
   | List of Type
   | Triv
val ` : 'a * ('a -> Type) -> Type
val generate_tools : string -> Type -> unit

end 


(* STRUCTURE / FUNCTOR BODIES *)

(* "Trivial" implementation of NAT using an inductive type with "intrinsic"
   coding operations. Not really the natural numbers, but much more efficient
   for most practical purposes *)

local

datatype nat = nat' of int | pair' of nat * nat 
             | inl' of nat | inr' of nat
             | list' of nat list

fun diverge () = diverge ()

structure Triv_Nat_IO : NAT_IO = struct
   type nat = nat
   exception Nat
   fun nat n = if n>=0 then nat' n else raise Nat
   fun int (nat' n) = n 
     | int _ = raise Nat
   fun str (nat' n) = Int.toString n
     | str (pair' (x,y)) = "("^str x^","^str y^")"
     | str (inl' x) = "?"^str x
     | str (inr' x) = 
          let val s = str x ;
              val i = substring (s,0,1)
          in if i = "?" orelse i = "!" then "!("^s^")" else "!"^s
          end
     | str (list' []) = "[]"
     | str (list' (x::t)) = "["^str x^concat (List.map str1 t)^"]"
   and str1 x = ","^str x
   exception NotImplemented
   fun ev s = raise NotImplemented
end

structure Triv_Nat_Pair : PAIR = struct
   type nat = nat
   fun pair x y = pair' (x,y)
   fun proj (pair' p) = p
     | proj _ = diverge ()
   fun fst x = #1 (proj x)
   fun snd x = #2 (proj x)
end

structure Triv_Nat_Cases : CASES = struct
   type nat = nat
   val inl = inl' and inr = inr'
   fun cases (inl' x) = (x,true)
     | cases (inr' x) = (x,false)
     | cases _ = diverge ()
   fun outlr x = #1 (cases x)
   fun isl (inl' x) = true
     | isl _ = false
end

structure Triv_Nat_Lst : LST = struct
   type nat = nat
   val empty = list' []
   fun cons x (list' l) = list' (x::l)
     | cons x _ = diverge ()
   exception Empty
   fun isEmpty (list' []) = true
     | isEmpty _ = false
   fun dest (list' []) = raise Empty
     | dest (list' (x::t)) = (x,list' t)
     | dest _ = diverge ()
   fun head x = #1 (dest x)
   fun tail x = #2 (dest x)
end

structure Triv_Nat : NAT = struct
   type nat = nat
   structure Nat_IO = Triv_Nat_IO
   structure Pair = Triv_Nat_Pair
   structure Cases = Triv_Nat_Cases
   structure Lst = Triv_Nat_Lst
end 

in

structure Triv_Nat_Utils :> NAT_UTILS = struct
   open Triv_Nat

   open Triv_Nat_IO
   val n0 = nat 0
   val n1 = nat 1

   open Triv_Nat_Lst 
   val codeList = list'
   fun decodeList (list' l) = l
     | decodeList _ = diverge ()

   open Triv_Nat_Cases
   val quest = inl
   val ans = inr
   val deQuest = outlr
   val deAns = outlr
   val isQuest = isl
   val isAns = not o isl
end

end 


(* Implementation in New Jersey SML of lambda algebra of lazy forests.
   This is the only bit that requires continuations. *)

functor Forest (
	structure Nat : NAT_UTILS
        ) : FOREST = struct

   open Nat Nat.Nat_IO Nat.Pair
   datatype forest = forest of nat -> (nat * forest)
   fun op% (forest f, n) = f n

(* Application operation *)

(* To apply a forest F to a forest G, we interpret the labels on F as 
   coding either *questions* to be put to some previously obtained
   subtree of G (identified by a timestamp), or *answers* giving the
   labels on the resulting tree. *)

   local
      exception Diverge
      fun diverge () = (print "Diverging...\n" ; raise Diverge)
      fun lookup t [] = diverge ()    (* diverge if timestamp is unknown *)
        | lookup t ((t',F)::rest) =
             if t=t' then F else lookup t rest

      fun play (forest f) previous timestamp n =
         case f n of (f_label,f_cont) =>
            if isAns f_label then
               (deAns f_label, 
                forest (play f_cont previous timestamp))
            else 
               case proj (deQuest f_label) of (n,t) =>
                  case (lookup (int t) previous) % n of (m,G) =>
                     play f_cont ((timestamp,G)::previous)
                          (timestamp+1) m
   in
   fun apply F G = forest (play F [(0,G)] 1)
   end


(* Abstraction operation. 
   Took me four weeks of intense concentration. Just don't ask how it works. *)

   open SMLofNJ.Cont
   type Lambda_Stamp = unit ref ;   

   (* Q-exceptions: restartable exceptions for generating question nodes *)

   local
      type Carries = nat * int
      type Gives   = nat * forest
      type Result  = Gives
      type Handler = Carries -> 
                     (Gives -> (unit -> Result) cont -> Result) -> Result
      type Exit = Lambda_Stamp * (unit -> Result) cont
      exception Q_E of Lambda_Stamp * Carries * Gives cont * Exit list
      val exit_list = ref ([] : Exit list) 

      local
         fun lookup s [] = (NONE,[])
           | lookup s ((s',k)::rest) =
                if s = s' then (SOME k,rest)
                else case lookup s rest of 
                   (v,rest') => (v,(s',k)::rest')
         fun extract_exit s =
            case lookup s (!exit_list) of 
               (v,exits) => (exit_list := exits ; v)
      in
      fun return_to s thunk =
         case extract_exit s of
             NONE => thunk
         | SOME k => throw k thunk

      fun Q_E_return (s,tuple as (s',x',k',pending)) =
         case extract_exit s of
             NONE => (fn () => raise Q_E tuple)
         | SOME k => throw k (fn () => raise Q_E (s',x',k',(s,k)::pending))
      end

   in 
   fun reset () =
      case !exit_list of
         [] => true
       |  _ => (exit_list := [] ; false)

   exception Q_exn of Lambda_Stamp * Carries * (Result -> Result)

   fun Q_try' stamp (code : unit -> Result) (handler : Handler) =
      let val answer = 
         code() handle (Q_E (s,x,k,pending)) =>
            if s <> stamp then raise Q_E (s,x,k,pending)
            else let fun resume y newExit = 
                    (exit_list := (stamp,newExit)::pending@(!exit_list) ;
                     callcc (fn k' => throw k y))
                 in (handler x resume)
                 end
      in
         return_to stamp (fn () => answer)
      end
      handle Q_exn tuple => return_to (#1 tuple) 
                            (fn () => raise Q_exn tuple)
           |   Q_E tuple => Q_E_return (stamp, tuple)
           |       other => return_to stamp (fn () => raise other)

   fun Q_try stamp code =
       Q_try' stamp code
          (fn t => fn C =>
              let fun D x = callcc (fn k => fn () => C x k) ()
              in raise Q_exn (stamp,t,D)
              end)
          ()

   fun Q_raise stamp x = 
      callcc (fn k => raise (Q_E (stamp,x,k,[])))

   end (* Q-exceptions *)

   (* A-exceptions: simple restartable exceptions allowing us to change
      our mind about the argument of a function in mid-computation. *)

   local
      type Carries = int
      type Gives   = forest
      type 'a Handler = Carries -> (Gives -> 'a) -> 'a
      exception A_E of Carries * Gives cont
   in
   fun A_try (code : unit -> 'a) (handler : 'a Handler) =
      code () handle (A_E (x,k)) =>
         handler x (throw k)

   fun A_raise x = 
      callcc (fn k => raise (A_E (x,k)))

   (* A little recursion to smuggle a forest F into the computation of F'.
      Timestamps are used to distinguish between different A_raise:s.

      Known inefficiency: we get nested calls to A_plugin accumulating 
      as we move down the forest (this happens even with lambda id).
      Getting rid of these should be possible, but not trivial. *)

   fun A_plugin t F F' =
      forest (fn m =>
         A_try 
            (fn () => 
                case F'%m of (i,F'') => (i, A_plugin t F F''))
            (fn t' => fn C => 
                if t'=t then C F else C (A_raise t')))

   end (* A-exceptions *)

   (* The abstraction operation itself *)

   fun lambda Phi =
      let val stamp = ref ()
          val Q_try = Q_try stamp
          val Q_raise = Q_raise stamp

          (* the main recursion *)
          fun lambda' timestamp Phi = 
             forest (fn m =>
                (case (Q_try (fn () => 
                   Phi (A_try 
                      (fn () => A_raise timestamp)
                      (fn t => fn C => 
                         if t=timestamp then
                            C (forest (fn n => Q_raise (n,t)))
                         else C (A_raise t)))      
                   % m))
                of (a, F') =>
                   (ans a, 
                    lambda' timestamp
                       (fn F => A_plugin timestamp F F')))
                handle Q_exn (tuple as (s,(n,timestamp'),C)) =>
                   if s <> stamp then raise Q_exn tuple
                   else (quest (pair n (nat timestamp')),
                         lambda' (timestamp+1)
                                 (fn F => forest (fn p => C(p,F)))))
      in
         lambda' 0 Phi
      end

end 


(* Irredundant collapse of a strategy: gets rid of repeated questions *)

functor Irred (
        structure Nat : NAT_UTILS
        structure Forest : FOREST
        sharing type Nat.nat = Forest.nat
        ) : IRRED = struct

exception Unknown_Timestamp 

local
   open Nat Nat.Nat_IO Nat.Pair Forest

   fun lookup_t t [] = raise Unknown_Timestamp
     | lookup_t t ((path,oldt,newt,value)::rest) =
         if t=oldt then (path,newt)
         else lookup_t t rest

   fun lookup_path path [] = NONE
     | lookup_path path ((path',oldt,newt,value)::rest) =
         if path = path' then SOME (newt,value)
         else lookup_path path rest

   fun irred' F history old_time new_time m =
      case F % m of (label,F') =>
         if isAns label then 
            (label, forest (irred' F' history old_time new_time))
         else 
            case proj (deQuest label) of (n,t) =>
               case lookup_t (int t) history of (path,newt) =>
                 (case lookup_path (n::path) history of
                     SOME (newt',value) => 
                        irred' F' ((n::path,old_time,newt',value)::history)
                                  (old_time+1) new_time value
                   | NONE =>
                       (quest (pair n (nat newt)),
                        forest (fn value =>
                           irred' F' 
                              ((n::path,old_time,new_time,value)::history)
                              (old_time+1) (new_time+1) value)))

in
type forest = forest
fun irred F = forest (irred' F [([],0,0,nat 0)] 1 1)
end

(* Also add the (linear) memoization operator here,
   once we've understood what's wanted? *)

end (* functor Irred *)


(* Retracts module: for representing general ML types as retracts of 
   type Forest. The corresponding translations for game plays are done
   by the Decoder and Encoder modules. *)

functor Retracts (
        structure Nat : NAT_UTILS
        structure Forest : FOREST
        sharing type Nat.nat = Forest.nat
        ) : RETRACTS = struct

local open Nat Forest in

type nat = nat
type forest = forest
type ('a,'b) retraction = ('a -> 'b) * ('b -> 'a)

infix oo
fun op oo ((f1,g1): ('b,'c) retraction, (f2,g2): ('a,'b) retraction) =
   (f1 o f2, g2 o g1) : ('a,'c) retraction

type 'a coding = ('a,nat) retraction
type 'a rep = ('a,forest) retraction

datatype ('a,'b) sum = left of 'a | right of 'b


(* Codings of flat types in nat *)

local open Nat Nat.Pair Nat.Cases in

val unit_coding : unit coding = 
    (fn () => n0, fn _ => ())
val bool_coding : bool coding = 
    (fn b => if b then n0 else n1, 
     fn n => (n = n0))
val int_coding  : int coding = 
    (fn i => if i>=0 then inl (Nat_IO.nat i) 
             else inr (Nat_IO.nat (~i-1)),
     fn n => case cases n of (n',b) =>
             if b then Nat_IO.int n' else ~(Nat_IO.int n'+1))
val nat_coding  : nat coding =
    (fn n => n, fn n => n)

fun product_coding ((f1,g1) : ''a coding) ((f2,g2) : ''b coding) =
    (fn (x,y) => pair (f1 x) (f2 y),
     fn n => case proj n of (n1,n2) => (g1 n1, g2 n2))
    : (''a * ''b) coding

(* we can easily do products of arity >2 if we need to. *)

fun sum_coding ((f1,g1) : ''a coding) ((f2,g2) : ''b coding) =
    (fn (left x) => inl (f1 x) | (right y) => inr (f2 y),
     fn n => case cases n of (n',b) => 
        if b then left (g1 n') else right (g2 n'))
    : (''a,''b) sum coding

fun list_coding ((f,g) : ''a coding) =
    (codeList o List.map f,
     List.map g o decodeList)
    : ''a list coding

end

(* Representations in forest.
   For the correct treatment of call-by-value types, we need to make some
   fiddly distinctions between ground types and higher types.
   This results in four variants of the arrow type constructor! *)

fun diverge () = diverge () 
exception Game_Over ;
val bottom_forest = forest (fn m => raise Game_Over) 
fun embed1 f = forest (fn m => (f m,bottom_forest))
fun project1 F = fn m => #1 (F % m)
fun thunk G = forest (fn m => (n0, G m))
fun eval F = fn m => #2 (F % m)

val nat_rep : nat rep =
    (fn n => embed1 (fn m => if m=n0 then n else diverge()),
     fn F => project1 F n0)

fun arrow0_rep ((f1,g1) : ''a coding) ((f2,g2) : ''b coding) =
    (fn h => embed1 (f2 o h o g1),
     fn F => g2 o project1 F o f1)
    : (''a -> ''b) rep

fun arrow1_rep ((f1,g1) : ''a coding) ((s2,t2) : 'b rep) =
    (fn h => thunk (s2 o h o g1),
     fn F => t2 o eval F o f1)
    : (''a -> 'b) rep

fun total_arrow_rep ((s1,t1) : 'a rep) ((s2,t2) : 'b rep) =
    (fn h => lambda (s2 o h o t1),
     fn F => t2 o (apply F) o s1)
    : ('a -> 'b) rep

fun arrow2_rep (R : 'a rep) (C : ''b coding) =
     (total_arrow_rep R (nat_rep oo C))
    : ('a -> ''b) rep

fun lift_rep (R : 'a rep) = 
    (arrow1_rep unit_coding R) : (unit -> 'a) rep

val arrow_decomp : ('a->'b, 'a->(unit->'b)) retraction =
    (fn f => fn x => (fn () => f x),
     fn g => fn x => g x ())

fun arrow3_rep (R1 : 'a rep) (R2 : 'b rep) =
    ((total_arrow_rep R1 (lift_rep R2)) oo arrow_decomp)
    : ('a -> 'b) rep

local open Nat.Cases in
fun product1_rep ((s1,t1) : 'a rep) ((s2,t2) : 'b rep) =
    (fn (x,y) => forest (fn m => 
        if isl m then s1 x % outlr m else s2 y % outlr m),
     fn F => (t1 (forest (fn m => F % inl m)),
              t2 (forest (fn m => F % inr m))))
    : ('a * 'b) rep
end

end
end (* functor Retracts *)


(* Decoder module: for interpreting plays in Forest in terms of the relevant
   ML types. (The Encoder module performs the inverse translation.)
   Used to display game plays intelligibly (i.e. in terms of the user's
   ML types) in execution traces. *)

functor Decoder (
        structure Nat : NAT_UTILS
        ) : DECODER = struct

local open Nat in

(* Plays and justification sequences *)

type nat = nat
datatype subtime = a | b
type time = int * subtime
fun timeToInt (t,_) = t
datatype QA = Q | A
fun QA_string Q = "Q" | QA_string A = "A"

datatype cell = cell of QA * nat * time * cell option ref
 (* times here are creation times of cells - follow pointers for 
    justification times *)
fun Q_cell (n,t,c_opt) = cell (Q,n,t,ref c_opt)
and A_cell (n,t,c) = cell (A,n,t, ref (SOME c))
exception Timestamp

(* Decompositions of sequences of moves corresponding to various 
   type constructors. Done dynamically using streams. 
   Plugging these stream operators together will yield the desired
   decoding operations. *)

type column = int
type j_pointer = time option
type move_info = QA * nat * column * time * j_pointer
type printer   = move_info -> string

datatype unraveller = 
   unraveller of cell -> move_info * unraveller

fun triv_unraveller col =
   unraveller (fn cell (L,n,t,r) =>
     ((L,n,col,t,
       case !r of NONE => NONE | SOME (cell(_,_,t',_)) => SOME t'),
      triv_unraveller col))

infix $
fun op$ (unraveller d, c) = d c

fun int' n = Nat_IO.int n handle _ => 0
fun destQuest n =
    case Pair.proj (deQuest n) of (n',t) => (n', int' t)
fun time_of NONE = NONE
  | time_of (SOME (cell (_,_,t,_))) = SOME t
(*
fun cell_of_time t [] = ref NONE
  | cell_of_time t ((c as cell (_,_,t',_))::rest) =
    if t=t' then ref (SOME c) else cell_of_time t rest
*)
fun cell_with_ts ts hist = 
    List.nth (hist, List.length hist-ts-1)
    handle Subscript => raise Timestamp

fun get_unraveller NONE _ = NONE
  | get_unraveller (SOME c) [] = NONE
  | get_unraveller (c_opt as SOME (cell (L,_,t,_))) 
               ((cell (L',_,t',_),k)::rest) =
       if t=t' andalso L=L' then SOME k 
       else get_unraveller c_opt rest

fun arrow0_unraveller col = 
   let fun dc (cell (Q,n,t,r)) = 
           ((Q, n, col, t, time_of (!r)), unraveller dc) 
         | dc (cell (A,n,t,r)) =
           ((A, n, col, t, time_of (!r)), unraveller dc) 
   in unraveller dc 
   end

fun arrow1_unraveller col D =
   let fun dc (cell (Q,n,t,r)) =
           ((Q, n, col, t, time_of (!r)), unraveller dc)
         | dc (c as cell (A,n,t,r)) =
           ((A, n0, col, t, time_of (!r)), unraveller (dc' t))
       and dc' t c' = 
           (case D $ c' of ((L,n,col,t',jp),D') =>
              ((L,n,col,t',SOME t),D'))               
   in unraveller dc
   end

(* For the others, it helps to decompose arrow into bang and linear arrow -
   also to decompose partial arrow into total arrow and lift *)

fun bang_unraveller D =
   let fun dc cell_unravellers (c as cell (L,n,t,r)) =
           let val E = case get_unraveller (!r) cell_unravellers of
                         NONE => D 
                       | SOME D' => D'
               val (info,E') = E $ c
           in (info, unraveller (dc ((c,E')::cell_unravellers)))
           end
   in unraveller (dc [])
   end

exception QA_error 

fun total_linear_arrow_unraveller D E =
   let fun dc D E hist (c as cell (Q,n,t,r)) = 
           (case E $ c of (info,E') =>
              (info, unraveller (dc' D E' c c hist)))
         | dc D E hist (c as cell (A,n,t,r)) =
           (case E $ c of (info,E') =>
              (info, unraveller (dc D E' hist)))
       and dc' D E pend last hist (cell (Q,n,t,r)) = 
           let val Ac = cell (A,n,t,ref (SOME last))
           in case D $ Ac of (info, D') =>
              (info, unraveller (dc' D' E pend Ac (Ac::hist)))
           end
         | dc' D E pend last hist (cell (A,n,t,r)) = 
           if isQuest n then
              let val (n',ts) = destQuest n
                  val Qc = cell (Q,n',t,ref (SOME (cell_with_ts ts hist)))
              in case D $ Qc of (info, D') =>
                   (info, unraveller (dc' D' E pend Qc hist))
              end
           else case E $ cell (A, deAns n, t, ref (SOME pend)) of 
                  (info,E') => (info, unraveller (dc D E' hist))
   in unraveller (fn (c as cell (Q,_,_,_)) => 
                  (case E $ c of (info,E') => 
                       (info, unraveller (dc' D E' c c [c])))
                | _ => raise QA_error)
   end

fun total_arrow_unraveller D E =
   total_linear_arrow_unraveller (bang_unraveller D) E

fun arrow2_unraveller col D =
   total_arrow_unraveller D (arrow0_unraveller col)

fun lift_unraveller col D = 
   arrow1_unraveller col D

fun arrow3_unraveller col D E =
   total_arrow_unraveller D (lift_unraveller col E)


(* Turning a raw play in the lambda algebra (alternating Q/A values) 
   into a stream of cells for consumption by an unraveller *)

datatype cell_stream = cell_stream of nat -> cell * cell_stream

local
fun simple' time NONE n = 
    let val Qc = Q_cell (n, (time,a), NONE)
    in (Qc, cell_stream (simple' time (SOME Qc))) end
  | simple' time (SOME (c as cell (Q,_,_,_))) n = 
    let val Ac = A_cell (n, (time,b), c)
    in (Ac, cell_stream (simple' (time+1) (SOME Ac))) end
  | simple' time (SOME (c as cell (A,_,_,_))) n =
    let val Qc = Q_cell (n, (time,a), SOME c)
    in (Qc, cell_stream (simple' time (SOME Qc))) end
in
val basic_stream = cell_stream (simple' 1 NONE)
end

(* Combining this with an unraveller to get an decoder *)

datatype decoder = decoder of nat -> move_info * decoder
fun op\ (decoder i, n) = i n

local
   fun make_decoder' (cell_stream cs) (unraveller d) =
      decoder (fn n =>
         let val (c,C') = cs n
             val (info,D') = d c
         in (info, make_decoder' C' D')
         end) 
in
fun make_decoder D = make_decoder' basic_stream D
end

end
end


(* Encoder module: for translating a typed game play into an untyped
   play in Forest (this is inverse to the Decoder translation).
   Used in interactive mode where the user plays moves in the typed game. *)

(* N.B. Error trapping for illegal pointers is incomplete. (E.g. visibility)
   Sometimes appears to accept a pointer but uses a different one. *)

functor Encoder (
        structure Nat : NAT_UTILS
        ) : ENCODER = struct

local open Nat in

type nat = nat
type column = int
type time = int
type j_pointer = time option
type move_info = nat * column * time * j_pointer
datatype encoder = 
   encoder of move_info -> nat * encoder
exception Column
exception Expired
exception Bad_Pointer
exception Wrong_Pointer

infix $
fun op$ (encoder e, tuple) = e tuple

type column_info = int * int * int

fun check_col col (left,this,right) =
   if col = this then () else raise Column
fun check_left col (left,this,right) =
   if left <= col andalso col < this then ()
   else raise Column
fun check_right col (left,this,right) = 
   if this <= col andalso col <= right then ()
   (* N.B. col=this is allowed *)
   else raise Column

fun check_ptr p t = 
    if p = SOME t then () else raise Wrong_Pointer

fun get_encoder NONE     _  = raise Bad_Pointer
  | get_encoder (SOME t) [] = raise Bad_Pointer
  | get_encoder (SOME t) ((t',F,ts)::rest) =
    if t=t' then (F,ts) else get_encoder (SOME t) rest

fun arrow0_encoder col_info = 
   encoder (fn (n,col,t,p) =>
     (check_col col col_info ;
      (* check pointer? *)
      (n, encoder (fn (n',col',t',p') =>
        (check_col col' col_info ;
         check_ptr p' t ;
         (n', encoder (fn _ => raise Expired)))))))

fun arrow1_encoder col_info E = 
   encoder (fn (n,col,t,p) =>
     (check_col col col_info ;
      (* check pointer? *)
      (n, encoder (fn (n',col',t',p') =>
         (check_col col' col_info ;
          check_ptr p' t ;
          (n', E))))))    (* or replace p' by NONE in first move? *)

fun bang_encoder E zerotime =
   let fun ec time_encoders timestamp (n,col,t,p) =
           let val (F,ts) = get_encoder p time_encoders
               val (n',F') = F $ (n,col,t,p)
           in (Pair.pair n' (Nat_IO.nat ts),
               encoder (ec' F' time_encoders timestamp))
           end
       and ec' F' time_encoders timestamp (n',col',t',p') =
          (* check_ptr p' t here has disastrous effects! *)
           let val (n'',F'') = F' $ (n',col',t',p')
           in (n'', encoder (ec ((t',F'',timestamp)::time_encoders) 
                               (timestamp+1)))
           end
   in encoder (ec [(zerotime,E,0)] 1)
   end

fun total_linear_arrow_encoder col_info E F =
   let fun O_right_ec E F (n,col,t,p) =
         (check_right col col_info ;
          case F$(n,col,t,p) of (n',F') =>
               (n', encoder (P_ec E F')))
       and O_left_ec E F (n,col,t,p) =
         (check_left col col_info ;
          case E$(n,col,t,p) of (n',E') =>
               (n', encoder (P_ec E' F)))
       and P_ec E F (n,col,t,p) =
          if col >= #2 col_info  (* if col is a right column *)
          then case F$(n,col,t,p) of (n',F') =>
                 (ans n', encoder (O_right_ec E F'))
          else case E$(n,col,t,p) of (n',E') =>
                 (quest n',
                  encoder (O_left_ec E' F))
   in encoder (O_right_ec E F)
   end

fun total_arrow_encoder col_info E F =
   encoder (fn (n,col,t,p) => 
      (total_linear_arrow_encoder col_info (bang_encoder E t) F) $ 
       (n,col,t,p))

fun arrow2_encoder col_info E =
   total_arrow_encoder col_info E (arrow0_encoder col_info)

fun lift_encoder col_info E =
   arrow1_encoder col_info E   (* with some check for 0? *)

fun arrow3_encoder (col_info as (left,this,right)) E F =
   total_arrow_encoder col_info E 
      (lift_encoder (this,this,right) F)

end
end


(* Stuff relating to syntax of ML types supported.
   Some rather fiddly distinctions between ground types and higher types
   are needed to treat call-by-value correctly. This is why we have
   three different versions of "arrow", for instance. *)

functor Types (
        structure Nat : NAT_UTILS 
        structure Decoder : DECODER
        structure Encoder : ENCODER
        sharing type Nat.nat = Decoder.nat = Encoder.nat
        ) : TYPES = struct  

type nat = Nat.nat

(* Syntax of types for user's purposes *)

datatype Type =
     Unit | Bool | Int | Nat 
   | Arrow of Type * Type
   | Times of Type * Type
   | Sum of Type * Type
   | List of Type
   | Triv

fun (x`f) = f x   

(* the user can write types in more or less ML syntax, e.g.
      (Int Arrow Int) Arrow (Int Times Bool) `List         *)

(* Syntax of types for internal purposes.
   Different kinds of arrow are distinguished ;
   also column information is incorporated. *)

datatype Type0 = 
     Unit0 | Bool0 | Int0 | Nat0 
   | Times0 of Type0 * Type0
   | Sum0 of Type0 * Type0
   | List0 of Type0
   | Triv0   (* used for "blank entries" that don't carry a value *)

type column = int

datatype Type1 =
     Arrow0 of Type0 * column * Type0
   | Arrow1 of Type0 * column * Type1
   | Arrow2 of Type1 * column * Type0
   | Arrow3 of Type1 * column * Type1
   | Triv1 of column
     (* Hope to extend these later!
        Even products are a bit subtle at higher types.
        Maybe make more use of "lift" type constructor? *)

type Type_data = Type1 * column

exception Type_Error

(* Converting a user type to an internal type *)

fun arrow_free (a Arrow b) = false
  | arrow_free (a Times b) = arrow_free a andalso arrow_free b
  | arrow_free (Sum(a,b))  = arrow_free a andalso arrow_free b
  | arrow_free (List a)    = arrow_free a 
  | arrow_free Triv        = false
  | arrow_free _           = true

fun type0 Unit = Unit0
  | type0 Bool = Bool0
  | type0 Int = Int0
  | type0 Nat = Nat0
  | type0 (a Times b) = Times0 (type0 a, type0 b)
  | type0 (Sum (a,b)) = Sum0 (type0 a, type0 b)
  | type0 (List a)    = List0 (type0 a)
  | type0 (a Arrow b) = raise Type_Error
  | type0 (Triv) = raise Type_Error
  
fun type_data a =
   let val col = ref (~1) ;
       fun next_col () = (col := !col + 1 ; !col)
       fun type1 (a Arrow b) =
           (case (arrow_free a, arrow_free b) of
               (true, true)   => Arrow0 (type0 a, next_col(), type0 b)
             | (true, false)  => Arrow1 (type0 a, next_col(), type1 b)
             | (false, true)  => Arrow2 (type1 a, next_col(), type0 b)
             | (false, false) => Arrow3 (type1 a, next_col(), type1 b))
         | type1 Triv = Triv1 (next_col())
         | type1 a =
           ((if arrow_free a then
                  print "Arrow type expected.\n"
             else print "Sorry! Type not yet supported.\n") ;
             raise Type_Error)
   in (type1 a, next_col()) : Type_data
   end

(* Given an annotated type, we compile a list of the flat types appearing in
   the columns (as questions/answers, or as opponent/player moves). *)

local 
   open Decoder 
   datatype OP = O | P
   fun neg O = P | neg P = O
in
fun flat_types_in (Arrow0 (a,col,b)) L =
    [(Q,neg L,col,a),(A,L,col,b)]
  | flat_types_in (Arrow1 (a,col,b)) L =
    [(Q,neg L,col,a),(A,L,col,Triv0)] @ flat_types_in b L
  | flat_types_in (Arrow2 (a,col,b)) L =
    flat_types_in a (neg L) @ [(Q,neg L,col,Triv0),(A,L,col,b)]
  | flat_types_in (Arrow3 (a,col,b)) L =
    flat_types_in a (neg L) @ [(Q,neg L,col,Triv0),(A,L,col,Triv0)] 
                            @ flat_types_in b L
  | flat_types_in (Triv1 col) L = []

fun types_in a = flat_types_in (#1(type_data a)) P
fun O_types_in a = List.map (fn (L,L',col,a) => (L,col,a))
                   (List.filter (fn (_,L,col,a) => L=O) (types_in a))
end

exception Unknown_Column

fun QA_lookup_type L col [] = raise Unknown_Column
  | QA_lookup_type L col ((L',_,col',a)::rest) =
    if L=L' andalso col=col' then a
    else QA_lookup_type L col rest


(* Generating source code from syntax trees for types *)

local fun S_expr' [] = ""
        | S_expr' [s] = s
        | S_expr' (s::t) = s^" "^S_expr' t
in fun S_expr l = "("^S_expr' l^")" end

fun ML_for_Type Unit = "unit"
  | ML_for_Type Bool = "bool"
  | ML_for_Type Int  = "int"
  | ML_for_Type Nat  = "nat"
  | ML_for_Type (a Arrow b) = 
      S_expr [ML_for_Type a, "->", ML_for_Type b]
  | ML_for_Type (a Times b) =
      S_expr [ML_for_Type a, "*", ML_for_Type b]
  | ML_for_Type (Sum (a,b)) =
      S_expr [ML_for_Type a, ",", ML_for_Type b] ^ " sum"
  | ML_for_Type (List a) =
      ML_for_Type a^" list"
  | ML_for_Type (Triv) = raise Type_Error

fun syntax_for_Type Unit = "Unit"
  | syntax_for_Type Bool = "Bool"
  | syntax_for_Type Int  = "Int"
  | syntax_for_Type Nat  = "Nat"
  | syntax_for_Type (a Arrow b) = 
      S_expr [syntax_for_Type a, "Arrow", syntax_for_Type b]
  | syntax_for_Type (a Times b) =
      S_expr [syntax_for_Type a, "Times", syntax_for_Type b]
  | syntax_for_Type (Sum (a,b)) =
      S_expr [syntax_for_Type a, ",", syntax_for_Type b] ^ " `Sum"
  | syntax_for_Type (List a) =
      syntax_for_Type a^" `List"
  | syntax_for_Type (Triv) = raise Type_Error

fun coding_for Unit0 = "unit_coding"
  | coding_for Bool0 = "bool_coding"
  | coding_for Int0  = "int_coding"
  | coding_for Nat0  = "nat_coding"
  | coding_for (Times0 (a,b)) = 
      S_expr ["product_coding", coding_for a, coding_for b]
  | coding_for (Sum0 (a,b)) =
      S_expr ["sum_coding", coding_for a, coding_for b]
  | coding_for (List0 a) =
      S_expr ["list_coding", coding_for a]
  | coding_for Triv0 = "unit_coding"

fun rep_for (Arrow0 (a,_,b)) =
      S_expr ["arrow0_rep", coding_for a, coding_for b]
  | rep_for (Arrow1 (a,_,b)) =
      S_expr ["arrow1_rep", coding_for a, rep_for b]
  | rep_for (Arrow2 (a,_,b)) =
      S_expr ["arrow2_rep", rep_for a, coding_for b]
  | rep_for (Arrow3 (a,_,b)) =
      S_expr ["arrow3_rep", rep_for a, rep_for b]
  | rep_for (Triv1 _) = raise Type_Error

fun rep_for_Type a =
    case type_data a of (a1,_) => rep_for a1



(* Conversion to strings, for ground types. Requires Nat. *)

local open Nat in
fun string0 Unit0 n = "()"
  | string0 Bool0 n = Bool.toString (n=n0)
  | string0 Int0 n  = 
      (case Cases.cases n of
          (n',true)  => Int.toString (Nat_IO.int n')
        | (n',false) => "~"^Int.toString (Nat_IO.int n'+1))
  | string0 Nat0 n  = Nat_IO.str n
  | string0 (Times0 (a,b)) n =
      (case Pair.proj n of (n1,n2) =>
          "(" ^ string0 a n1 ^ "," ^ string0 b n2 ^ ")")
  | string0 (Sum0 (a,b)) n =
      (case Cases.cases n of
          (n1,true)  => "left("  ^ string0 a n1 ^ ")"
        | (n2,false) => "right(" ^ string0 b n2 ^ ")")
  | string0 (List0 a) n =
      let fun restOf [] = "]"
            | restOf (h::t) = "," ^ string0 a h ^ restOf t
      in
         case decodeList n of 
              [] => "[]"
            | h::t => "[" ^ string0 a h ^ restOf t
      end
  | string0 Triv0 n = "-"
end

(* Constructing an appropriate decoder for a given type. *)

local 
   open Decoder
   exception No_Triv
   fun unraveller_for (Arrow0 (a,col,b)) =
       arrow0_unraveller col
     | unraveller_for (Arrow1 (a,col,b)) =
       arrow1_unraveller col (unraveller_for b)
     | unraveller_for (Arrow2 (a,col,b)) =
       arrow2_unraveller col (unraveller_for a)
     | unraveller_for (Arrow3 (a,col,b)) =
       arrow3_unraveller col (unraveller_for a) (unraveller_for b)
     | unraveller_for (Triv1 col) = raise No_Triv
in
type decoder = decoder
type move_info = move_info
type printer = move_info -> string
type QA = QA
val op\ = op\
fun decoding a =
    case type_data a of (a1,cols) => 
       (make_decoder (unraveller_for a1), cols,
        case types_in a of catalogue =>
        fn (L,n,col,t,p) => 
           (QA_string L ^ " " ^ string0 (QA_lookup_type L col catalogue) n))
end

(* Constructing an appropriate encoder for a given type.
   Only the "arrow structure" of the type is dealt with here;
   the coding of flat types into nat is added by the generated code. *)

exception Bad_Column

local
   open Encoder
   fun encoder_for (Arrow0 (a,col,b)) left right =
          arrow0_encoder (left,col,right)
     | encoder_for (Arrow1 (a,col,b)) left right =
          arrow1_encoder (left,col,right)
             (encoder_for b (col+1) right)
     | encoder_for (Arrow2 (a,col,b)) left right =
          arrow2_encoder (left,col,right)
             (encoder_for a left (col-1))
     | encoder_for (Arrow3 (a,col,b)) left right =
          arrow3_encoder (left,col,right)
             (encoder_for a left (col-1))
             (encoder_for b (col+1) right)
     | encoder_for (Triv1 col) _ _ = raise Bad_Column (* to be added *)

in
type encoder = encoder
type e_move_info = Encoder.move_info
val op$ = op$
fun encoding a =
   case type_data a of (a1,cols) =>
      encoder_for a1 0 (cols-1)
end

end 


(* Front-end stuff, for pretty-printing execution traces and
   also playing interactive games. *)

(* Version 1.1: functor arguments rewritten to compile under NJSML 110.42.
   Thanks to Ethan Aubin for help with this. *)

functor Display (
       structure Forest  : FOREST
       structure Decoder : DECODER
       structure Types   : TYPES where type printer = Decoder.printer
       structure Encoder : ENCODER

       sharing type Forest.nat = Types.nat = Decoder.nat = Encoder.nat
       sharing type Types.decoder = Decoder.decoder
(*     sharing type Types.printer = Decoder.printer *)
       sharing type Types.encoder = Encoder.encoder
       ) : DISPLAY = struct

(* formatting parameters *)
structure Format = struct
   val column_width = ref 10
   val left_margin  = ref 10
   val pointer_margin = ref 2
   val blank_lines = ref true
end ;

local
   open Format
   fun spaces 0 = "" 
     | spaces n = if n<0 then "  " else " " ^ spaces (n-1) 
   fun make_up_length s k = 
       s ^ spaces (k - String.size s)
       
   fun print_line this other flag cols (s,col,t,jp) =
       print (make_up_length ("  "^this^Int.toString t^":") (!left_margin) ^
              make_up_length (spaces (col * !column_width)^s) 
                             (cols * !column_width + !pointer_margin) ^
              (case jp of NONE => "(init)"
                        | SOME j => "(by "^other^Int.toString j^")") ^ 
              (if flag then "\n\n" else "\n"))
   fun O_print cols stuff = print_line "O" "P" false cols stuff
   and P_print cols stuff = print_line "P" "O" (!blank_lines) cols stuff
   open Forest Types Encoder Decoder
in

type nat = nat
type forest = forest
type Type = Type
type 'a rep = ('a -> forest) * (forest -> 'a)

fun output_info (info as (L,n,col,t,p)) printer = 
   (printer info, col, timeToInt t, Option.map timeToInt p)

fun trace_forest (dec,cols,printer) (forest f) =
   forest (fn m =>
      let val (info, dec') = dec \ m
      in (O_print cols (output_info info printer) ;
          let val (n,F') = f m
              val (info', dec'') = dec' \ n
          in  P_print cols (output_info info' printer) ;
              (n, trace_forest (dec'',cols,printer) F')
          end)
      end)

(* stuff for displaying a type in order to show the meaning of the columns *)

local 
   val size = String.size
   and substring = String.substring
   and extract = String.extract
   and explode = String.explode
   fun count_arrows s =
      if size s < 2 then 0
      else if substring (s,0,2) = "->" 
           then 1+count_arrows (extract (s,2,NONE))
           else count_arrows (extract (s,1,NONE))
   exception No_Arrow
   fun pos_of_arrow s =
      if size s < 2 then raise No_Arrow
      else if substring (s,0,2) = "->" then 0
      else 1 + pos_of_arrow (extract (s,1,NONE))
   fun last_close_br [] br_count last curr_pos = last
     | last_close_br (#"("::rest) br_count last curr_pos =
          last_close_br rest (br_count+1) last (curr_pos+1)
     | last_close_br (#")"::rest) br_count last curr_pos =
          if br_count > 0
          then last_close_br rest (br_count-1) last (curr_pos+1)
          else last_close_br rest 0 (SOME curr_pos) (curr_pos+1)
     | last_close_br (_::rest) br_count last curr_pos =
          last_close_br rest br_count last (curr_pos+1)
   fun find_last_close_br s =
          last_close_br (explode s) 0 NONE 1
   fun segment s =
      if count_arrows s < 2 then [s]
      else let val a1 = pos_of_arrow s + 2
               val a2 = pos_of_arrow (extract (s,a1,NONE))
           in case find_last_close_br (extract (s,a1,SOME a2)) of
               NONE   => substring (s,0,a1) :: 
                         segment (extract (s,a1,NONE))
             | SOME b => substring (s,0,a1+b) ::
                         segment (extract (s,a1+b,NONE))
           end
   fun segment1 s =
      if count_arrows s = 0 then [s]
      else case pos_of_arrow s of a1 =>
         substring (s,0,a1) :: segment (extract (s,a1,NONE))
   fun spaces 0 = "" | spaces n = " " ^ spaces (n-1)
   val begin_line = " "  (* shifts everything one character to right *)
   fun display_first_segment s =
       if size s <= !left_margin 
       then print (spaces (!left_margin - size s) ^ s)
       else print (s ^ "\n" ^ begin_line ^ spaces (!left_margin))
   fun display_type' [] hpos colpos = print "\n"
     | display_type' (s::rest) hpos colpos =
      let val leftpos = colpos - pos_of_arrow s in
         (if hpos <= leftpos then print (spaces (leftpos - hpos) ^ s)
          else print ("\n" ^ begin_line ^ spaces leftpos ^ s)) ;
          display_type' rest (leftpos + size s) (colpos + !column_width)
      end
   fun display_column_numbers i n =
      if i=n then print (if !Format.blank_lines then "\n\n" else "\n")
      else case Int.toString i of s =>
         (print (s ^ spaces (!column_width - String.size s)) ;
          display_column_numbers (i+1) n)
in
fun display_type (a : Type) =
    case segment1 (Types.ML_for_Type a) of
         [] => ()
      | (s::rest) =>
           (print begin_line ;
            display_first_segment s ;
            display_type' rest (!left_margin) (!left_margin) ;
            print (spaces (!left_margin + 1)) ;
            display_column_numbers 0 (List.length rest))
end

fun trace_strategy (s,t) (a : Type) x =
   (print "  Trace initiated for type: \n" ;
    display_type a ;
    (t o trace_forest (decoding a) o s) (x))
   (* user must ensure that `a' is the syntax tree for the type 'a. *)


(* Stuff for interactive mode, managing encodings and decodings.
   Supports backup. *)

fun column_error (col,p) =
   print ("  Illegal move: column " ^ Int.toString col ^ " not enabled " ^ 
          (case p of NONE   => "initially.\n"
                  | SOME q => "by P"^Int.toString q^".\n"))

fun expired_error (col,p) =
   print ("  Illegal move: subgame at P" ^
          Int.toString (case p of NONE => 0 | SOME q => q) ^
          " has expired.\n")

fun bad_pointer_error (col,p) =
   print ("  Illegal move: bad pointer.\n")

fun wrong_pointer_error (col,p) =
   print ("  Illegal move: wrong pointer.\n")

fun not_justified_error col = 
   print ("  Illegal move: no justifier available for column " ^
          Int.toString col ^ ".\n")

local
   val enc = ref (encoding (Nat Arrow Nat))       (* dummy *)
   val dec = ref (#1 (decoding (Nat Arrow Nat))) (* dummy *)
   val state = ref (Forest.lambda (fn f => f))    (* dummy *)
   val init_state = ref (!state)
   val time  = ref 0
   val columns = ref 0
   val printer = ref (fn _ => "")
   val type_a = ref Nat
   val backup_list = ref ([] : (forest * encoder * decoder * 
                                (nat * int * int option)) list)
   exception Not_Justified
in
fun start_interaction (s,t) (a : Type) x = 
   (print "  Starting game for type: \n" ;
    display_type a ;
    type_a := a ;
    enc := encoding a ;
    case decoding a of (I,cols,pr) =>
       (dec := I ; columns := cols ; printer := pr) ;
    state := s x ;
    init_state := s x ;
    time := 1 ;
    backup_list := [])
fun play_move (n,col,p) =
    let val (n',E') = !enc $ (n,col,!time,p) ;
        val (info,I') = !dec \ n' ;
        val (m,F') = !state % n' ;
        val (info' as (L1,n1,col1,t1,p1),I'') = I' \ m ;
        val (m',E'') = E' $ (n1,col1,timeToInt t1,Option.map timeToInt p1)
    in
        O_print (!columns) (output_info info (!printer)) ;
        P_print (!columns) (output_info info' (!printer)) ;
        backup_list := (!state,!enc,!dec,(n,col,p)) :: !backup_list ;
        enc := E'' ; dec := I'' ; state := F' ; time := !time+1 
    end
    handle Encoder.Column  => column_error (col,p)
         | Encoder.Expired => expired_error (col,p)
         | Encoder.Bad_Pointer   => bad_pointer_error (col,p)
         | Encoder.Wrong_Pointer => wrong_pointer_error (col,p)
fun find_justifier 0 (n,col) = 
    ((!enc $ (n,col,!time,NONE) ; NONE)
    handle _ => raise Not_Justified)
  | find_justifier t (n,col) =
    (!enc $ (n,col,!time,SOME t) ; SOME t)
    handle _ => find_justifier (t-1) (n,col)
fun play_auto_move (n,col) =
    play_move (n,col,find_justifier (!time-1) (n,col))
    handle Not_Justified => not_justified_error (col)
fun backup b =
    if b >= length (!backup_list) then
    (* back up to start of game *)
    case !type_a of a =>
       (print "  Restarting game for type: \n" ;
        display_type a ;
        enc := encoding a ;
        case decoding a of (I,cols,pr) =>
           (dec := I ; columns := cols ; printer := pr) ;
        state := !init_state ;
        time := 1 ;
        backup_list := [])
    else case List.nth (!backup_list,b) of (S,E,D,info) =>
       (backup_list := List.drop (!backup_list,b+1) ;
        state := S ; enc := E ; dec := D ; time := !time - b - 1 ;
        play_move info)
end

structure Interact_Tools = struct
   val backup = backup
   val init = NONE : int option
   val by   = SOME : int -> int option
end

end
end


(* Code generator: generates and then reads in a little bit of source code
   specific to the type of the user's program. *)

functor Code_Gen (
        structure Types   : TYPES
        structure Decoder : DECODER
        sharing type Types.QA = Decoder.QA
        ) : CODE_GEN = struct
local open TextIO in

open Types

fun generate_tools name ty =
   let val _ = rep_for_Type ty     (* first check to see if ty is supported *)
       val filename = name^".sml"
       val out = openOut filename
       fun write s = output (out,s^"\n")
       fun newline () = write ""
   in
       print ("Writing file \""^filename^"\"...\n") ;
       write ("(* File: "^filename^" *)") ;
       write ("(* Generated by Stratagem (Version 1.0) on "^
              Date.toString (Date.fromTimeLocal (Time.now ()))^
              " *)\n") ;
       newline () ;
       write ("structure "^name^" = struct ") ;
       (* add signature constraint ": TOOLS" ? *)
       write ("local open Retracts Types Display in") ;
       newline () ;
       write ("type the_type = "^ML_for_Type ty) ;
       newline () ;
       write ("val retraction = "^rep_for_Type ty) ;
       write  "            : the_type rep" ;
       newline () ;
       write ("val syntax_tree = "^syntax_for_Type ty) ;
       write ("fun show_type () = Display.display_type syntax_tree") ;
       newline () ;
       write ("val trace = (trace_strategy retraction syntax_tree)") ;
       write  "            : the_type -> the_type" ;
       newline () ;
       write ("val start_game = start_interaction retraction syntax_tree") ;
       newline () ;
       List.map (fn (L,col,a) => 
              (write ("fun " ^ Decoder.QA_string L ^ Int.toString col ^
                      " x = play_auto_move (#1 " ^ coding_for a ^ " x," ^
                      Int.toString col ^ ")") ;
               write ("fun " ^ Decoder.QA_string L ^ Int.toString col ^
                      "' x p = play_move (#1 " ^ coding_for a ^ " x," ^
                      Int.toString col ^ ",p)"))) 
           (O_types_in ty) ;
       newline () ;
       write ("val optimize = " ^ 
              "(#2 retraction) o Irred.irred o (#1 retraction)") ;
       newline () ;
       write "open Display.Interact_Tools" ;
       newline () ;
       write ("end end ; \n") ;
       closeOut out ;
       use filename
   end
end
end 


(* BUILD COMMANDS *)

structure Nat = Triv_Nat_Utils

structure Forest = 
   Forest (structure Nat = Nat)

structure Irred =
   Irred (structure Nat = Nat
         and Forest = Forest)

structure Retracts = 
   Retracts (structure Nat = Nat
            and Forest = Forest)

structure Decoder = 
   Decoder (structure Nat = Nat)

structure Encoder = 
   Encoder (structure Nat = Nat)

structure Types = 
   Types (structure Nat = Nat and Decoder = Decoder
          and Encoder = Encoder)

structure Display =
   Display (structure Forest = Forest and Types = Types
            and Decoder = Decoder and Encoder = Encoder)

structure Code_Gen =
   Code_Gen (structure Types = Types and Decoder = Decoder) 

(* OS.FileSys.chDir "Generated" ; *)
open Code_Gen 
val chDir = OS.FileSys.chDir
val reset = Forest.reset 

(* END OF FILE *)

