(*********************************************************************)
(*                                                                   *)
(*                      STRATAGEM Version 1.1                        *)
(*                      Complete source code                         *)
(*                                                                   *)
(*  (C) Daniel Hillerström, The University of Edinburgh, June 2021   *)
(*                      Version 1.1: June 2021                       *)
(*                                                                   *)
(*********************************************************************)
(*                                                                   *)
(* Complete OCaml source code for the STRATAGEM system, Version 1.1. *)
(* Contains signature declarations, functor bodies, build commands.  *)
(* OCaml 5.00.                                                       *)
(* See the accompanying file "userdoc.txt" for background and        *)
(* user documentation.                                               *)
(*                                                                   *)
(*********************************************************************)

(* An implementation of Call/cc; temporary structure to preserve
   familiarity with SML/NJ code. *)
module Callcc: sig
  type 'a cont
  val throw : 'a cont -> 'a -> 'b
  val callcc : ('a cont -> 'a) -> 'a
  val prompt : (unit -> 'a) -> 'a
end = struct
  type empty = |
  type 'a cont = 'a -> empty
  type _ Effect.t += Callcc : (('a -> 'b) -> 'a) -> 'a Effect.t

  let throw : 'a cont -> 'a -> 'b
    = fun k x -> match k x with | _ -> .

  let callcc : (('a -> 'b) -> 'a) -> 'a
    = fun f -> Effect.perform (Callcc f)

  let rec hprompt : unit -> ('a, 'a) Effect.Deep.handler
    = fun () ->
    let open Effect.Deep in
      { retc = (fun ans -> ans)
      ; exnc = raise
      ; effc = (fun (type a) (eff : a Effect.t) ->
        match eff with
        | Callcc f ->
           Some (fun (k : (a, _) continuation) ->
               let exception Throw of a in
               let cont : a -> empty
                 = fun x -> raise (Throw x)
               in
               try
                 prompt (fun () ->
                     let ans = f (fun x -> match cont x with _ -> .) in
                     continue k ans)
               with
               | Throw x -> continue k x)
        | _ -> None) }
  and prompt : (unit -> 'a) -> 'a
    = fun f -> Effect.Deep.match_with f () (hprompt ())
end

(* SIGNATURE DECLARATIONS *)

(* Natural numbers with coding functions.
   Actually, we only require an infinite set X equipped with
   codings for X*X, X+X, X list, and an injection int -> X. *)

module type NAT_IO = sig         (* conversion between natural numbers *)
    type nat                     (* and other types *)
    val nat : int -> nat
    val int : nat -> int
    val str : nat -> string
    val ev : string -> nat
    exception NatErr
end

module type PAIR = sig           (* bijective coding NxN <-> N *)
  type nat
  val pair : nat -> nat -> nat
  val proj : nat -> (nat * nat)
  val fst : nat -> nat           (* implementations of fst,snd may be *)
  val snd : nat -> nat           (* faster than that of proj *)
end

module type CASES = sig          (* bijective coding N+N <-> N *)
  type nat
  val inl : nat -> nat
  val inr : nat -> nat
  val cases : nat -> (nat * bool)
  val outlr : nat -> nat
  val isl : nat -> bool
end

module type LST = sig            (* bijective coding List[N] <-> N *)
  type nat
  val empty : nat
  val cons : nat -> nat -> nat
  val dest : nat -> (nat * nat)
  val head : nat -> nat
  val tail : nat -> nat
  val is_empty : nat -> bool
  exception Empty
end

module type NAT = sig
  type nat
  module Nat_IO : NAT_IO with type nat = nat
  module Pair : PAIR with type nat = nat
  module Cases : CASES with type nat = nat
  module Lst : LST with type nat = nat
end

(* Enhanced version including some utilities definable from the above *)

module type NAT_UTILS = sig
  include NAT
  val n0 : nat
  val n1 : nat
  val code_list : nat list -> nat
  val decode_list : nat -> nat list
  val quest : nat -> nat
  val ans : nat -> nat
  val de_quest : nat -> nat
  val de_ans : nat -> nat
  val is_quest : nat -> bool
  val is_ans : nat -> bool
end

(* Lambda algebra of lazy forests *)

module type FOREST = sig
   type nat
   type forest = Forest of (nat -> (nat * forest))
   val (%)    : forest -> nat -> nat * forest
   val apply  : forest -> forest -> forest
   val lambda : (forest -> forest) -> forest
   val reset  : unit -> bool
   module A: sig
    (* type carries = int *)
    (* type gives = forest *)
    (* type 'a handler = carries -> (gives -> 'a) -> 'a *)
     val a_try : (unit -> 'a) -> (int -> (forest -> 'a) -> 'a) -> 'a
   end
end

module type IRRED = sig
  type forest
  val irred : forest -> forest
end

(* Stuff for representing general ML types as retracts of forest. *)

module type RETRACTS = sig
  type nat
  type forest
  type ('a, 'b) sum = Inl of 'a
                    | Inr of 'b
  type ('a, 'b) retraction = ('a -> 'b) * ('b -> 'a)
  type 'a coding = ('a, nat) retraction
  type 'a rep = ('a, forest) retraction

  val unit_coding : unit coding
  val bool_coding : bool coding
  val int_coding : int coding
  val nat_coding : nat coding
  val product_coding : 'a coding -> 'b coding -> ('a * 'b) coding
  val sum_coding : 'a -> 'b coding -> ('a, 'b) sum coding
  val list_coding : 'a coding -> 'a list coding

  val arrow0_rep : 'a coding -> 'b coding -> ('a -> 'b) rep
  val arrow1_rep : 'a coding -> 'b rep -> ('a -> 'b) rep
  val arrow2_rep : 'a rep -> 'b coding -> ('a -> 'b) rep
  val arrow3_rep : 'a rep -> 'b rep -> ('a -> 'b) rep
end

(* Stuff for interpreting forest dialogues as game plays for general
   ML types. Closely related to RETRACTS. *)

module type DECODER = sig
  type unraveller
  val arrow0_unraveller : int -> unraveller
  val arrow1_unraveller : int -> unraveller -> unraveller
  val arrow2_unraveller : int -> unraveller -> unraveller -> unraveller
  val arrow3_unraveller : int -> unraveller -> unraveller -> unraveller

  type nat
  type time
  val int_of_time : time -> int
  type qa = Q | A
  val string_of_qa : qa -> string
  type move_info = qa * nat * int * time * time option

  type printer = move_info -> string

  type decoder

  val make_decoder : unraveller -> decoder

  val (//) : decoder * nat -> move_info * decoder
end

(* The other direction: stuff for translating a game play for the ML type
   into a forest dialogue. *)

module type ENCODER = sig
  type nat
  type encoder
  type move_info = nat * int * int * int option
  val ($) : encoder * move_info -> nat * encoder

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

module type TYPES = sig
  type nat
  type typ = Unit
           | Bool
           | Int
           | Nat
           | Arrow of typ * typ
           | Times of typ * typ
           | Sum of typ * typ
           | List of typ
           | Triv (* for debugging *)

  exception Type_Error
  val tick : 'a * ('a -> typ) -> typ
  val ml_for_type : typ -> string
  val syntax_for_type : typ -> string
  val rep_for_type : typ -> string

  type decoder
  type encoder
  type printer
  val decoding : typ -> decoder * int * printer
  val encoding : typ -> encoder

  type typ0
  type qa
  val coding_for : typ0 -> string
  val o_types_in : typ -> (qa * int * typ0) list
end

(* Displaying game traces *)
module type DISPLAY = sig
  type nat
  type forest
  type typ
  type 'a rep = ('a -> forest) * (forest -> 'a)

  val display_type : typ -> unit
  val trace_strategy : 'a -> 'a rep -> typ -> 'a -> 'a
  val start_interaction : 'a rep -> typ -> 'a -> unit
  val play_move : nat * int * int option -> unit
  val play_auto_move : nat * int -> unit

  module Interact_Tools: sig
    val backup : int -> unit
    val init : int option
    val by : int -> int option
  end

  (* formatting parameters *)
  module Format: sig
    val column_width : int ref
    val left_margin : int ref
    val pointer_margin : int ref
    val blank_lines : bool ref
  end
end

(* Stuff for generating and loading ML code specific to the type
   specified by the user *)
module type CODE_GEN = sig
  type typ = Unit
           | Bool
           | Int
           | Nat
           | Arrow of typ * typ
           | Times of typ * typ
           | Sum of typ * typ
           | List of typ
           | Triv

  val tick : 'a * ('a -> typ) -> typ
  val generate_tools : string -> typ -> unit
end

(* STRUCTURE / FUNCTOR BODIES *)
type nat = Nat of int
         | Pair of nat * nat
         | Inl of nat
         | Inr of nat
         | List of nat list

let rec diverge () = diverge ()

module Triv_Nat_IO : NAT_IO with type nat = nat = struct
  type nonrec nat = nat

  exception NatErr

  let nat n = if n >= 0
              then Nat n
              else raise NatErr

  let int = function
    | Nat n -> n
    | _ -> raise NatErr

  let rec str = function
    | Nat n -> string_of_int n
    | Pair (x, y) -> "(" ^ str x ^ "," ^ str y ^ ")"
    | Inl x -> "?" ^ str x
    | Inr x ->
       let s = str x in
       let i = String.sub s 0 1 in
       if i = "?" || i = "!" then "!(" ^ s ^ ")" else "!" ^ s
    | List [] -> "[]"
    | List (x :: xs) -> "[" ^ str x ^ String.concat "" (List.map (fun x -> "," ^ str x) xs) ^ "]"

  exception NotImplemented
  let ev _ = raise NotImplemented
end

module Triv_Nat_Pair : PAIR with type nat = nat = struct
  type nonrec nat = nat

  let pair x y = Pair (x, y)
  let proj = function
    | Pair (x,y) -> (x,y)
    | _ -> diverge ()
  let fst x = fst (proj x)
  let snd x = snd (proj x)
end

module Triv_Nat_Cases : CASES with type nat = nat = struct
  type nonrec nat = nat

  let inl x = Inl x
  let inr x = Inr x

  let cases = function
    | Inl x -> (x, true)
    | Inr y -> (y, false)
    | _ -> diverge ()

  let outlr x = fst (cases x)
  let isl = function
    | Inl _ -> true
    | _ -> false
end

module Triv_Nat_Lst : LST with type nat = nat = struct
  type nonrec nat = nat

  let empty = List []
  let cons x = function
    | List xs -> List (x :: xs)
    | _ -> diverge ()

  exception Empty

  let is_empty = function
    | List [] -> true
    | _ -> false

  let dest = function
    | List [] -> raise Empty
    | List (x :: xs) -> (x, List xs)
    | _ -> diverge ()

  let head x = fst (dest x)
  let tail x = snd (dest x)
end

module Triv_Nat : NAT with type nat = nat = struct
  type nonrec nat = nat
  module Nat_IO = Triv_Nat_IO
  module Pair = Triv_Nat_Pair
  module Cases = Triv_Nat_Cases
  module Lst = Triv_Nat_Lst
end

module Triv_Nat_Utils : NAT_UTILS with type nat = nat = struct
  include Triv_Nat

  open Triv_Nat_IO
  let n0 = nat 0
  let n1 = nat 1

  open Triv_Nat_Lst
  let code_list x = List x
  let decode_list = function
    | List xs -> xs
    | _ -> diverge ()

  open Triv_Nat_Cases
  let quest = inl
  let ans = inr
  let de_quest = outlr
  let de_ans = outlr
  let is_quest = isl
  let is_ans x = not (isl x)
end

module Forest(Nat : NAT_UTILS with type nat = nat): sig
  include FOREST with type nat = nat
end = struct
  type nonrec nat = nat
  open Nat
  open Nat.Nat_IO
  open Nat.Pair

  type forest = Forest of (nat -> (nat * forest))

  let run (Forest f) n = f n
  let (%) = run
  (* Application operation *)

  (* To apply a forest F to a forest G, we interpret the labels on F
     as coding either *questions* to be put to some previously
     obtained subtree of G (identified by a timestamp), or *answers*
     giving the labels on the resulting tree. *)

  (* let apply : forest -> forest -> forest *)
  (*   = let rec apply f g = Forest (fun i -> apply'' (f % i) g) *)
  (*     and apply'' (n, f) g = (n, apply f g) in *)
  (*     apply *)

  (* Much harder is the "lambda" operation, which requires
     catchcont3. Some "error handling" is needed here to cope with
     non-linear behaviour where it can't arise. *)

  (* type _ Effect.t += Question : nat -> nat Effect.t *)
  (* let question n = Effect.perform (Question n) *)

  (* type branch = nat -> nat * forest *)
  (* exception NonLinearError *)
  (* let error_branch : branch *)
  (*   = fun _i -> raise NonLinearError *)

  (* let lambda : (forest -> forest) -> forest *)
  (*   = let rec lambda phi = lambda' (fun f -> phi (Forest f)) *)
  (*     and lambda' (p : branch -> forest) = *)
  (*       Forest (fun h -> lambda'' (fun g -> p g % h)) *)
  (*     and lambda'' (_r : branch -> nat * forest) = *)
  (*       failwith "TODO" *)
  (*     in *)
  (*     lambda *)

  (* let reset = failwith "TODO" *)

  let apply f g =
    let exception Diverge in
    let diverge () = print_endline "Diverging..."; raise Diverge in
    let rec lookup t = function
      | [] -> diverge () (* diverge if timestamp is unknown *)
      | (t', f) :: rest ->
         if t = t' then f else lookup t rest
    in
    let rec play (Forest f) previous timestamp n =
      let (f_label, f_cont) = f n in
      if is_ans f_label then (de_ans f_label, Forest (play f_cont previous timestamp))
      else let (n, t) = proj (de_quest f_label) in
           let (m, g) = run (lookup (int t) previous) n in
           play f_cont ((timestamp, g) :: previous) (timestamp+1) m
    in
    Forest (play f [(0, g)] 1)

  (* Abstraction operation *)
  type lambda_stamp = unit ref

  (* Q-exceptions: restartable exceptions for generating question nodes *)
  module Q = struct
    type 'a cont = 'a Callcc.cont
    type carries = nat * int
    type gives = nat * forest
    type result = gives
    type handler = carries -> (gives -> (unit -> result) cont -> result) -> result
    type exit = lambda_stamp * (unit -> result) cont

    let exit_list = ref ([] : exit list)
    exception Q_E of lambda_stamp * carries * gives cont * exit list
    let rec lookup s = function
      | [] -> (None, [])
      | (s', k) :: rest ->
         if s = s' then (Some k, rest)
         else match lookup s rest with
              | (v, rest') -> (v, (s', k) :: rest')

    let extract_exit s =
      match lookup s !exit_list with
      | (v, exits) -> exit_list := exits; v

    let return_to s f =
      match extract_exit s with
      | None -> f
      | Some k -> Callcc.throw k f

    let q_e_return (s, (s', x', k', pending)) =
      match extract_exit s with
      | None -> (fun () -> raise (Q_E (s', x', k', pending)))
      | Some k -> Callcc.throw k (fun () -> raise (Q_E (s', x', k', (s, k) :: pending)))

    let reset () =
      match !exit_list with
      | [] -> true
      | _ -> (exit_list := []; false)


    exception Q_exn of lambda_stamp * carries * (result -> result)

    let q_try' : lambda_stamp -> (unit -> result) -> handler -> (unit -> result)
      = fun stamp code handler ->
      let answer =
        try code () with
        | Q_E (s, x, k, pending) ->
           if s <> stamp
           then raise (Q_E (s, x, k, pending))
           else let resume y new_exit =
                  let open Callcc in
                  exit_list := (stamp, new_exit) :: pending @ !exit_list;
                  callcc (fun _k' -> throw k y)
                in handler x resume
      in
      try
        return_to stamp (fun () -> answer)
      with
      | Q_exn (s, x, k) ->
         return_to s (fun () -> raise (Q_exn (s, x, k)))
      | Q_E (s, x, k, pending) -> q_e_return (s, (s, x, k, pending))
      | other -> return_to stamp (fun () -> raise other)

    let q_try : lambda_stamp -> (unit -> result) -> result
      = fun stamp code ->
      q_try' stamp code
        (fun t c ->
          let open Callcc in
          let d x = callcc (fun k () -> c x k) () in
          raise (Q_exn (stamp, t, d))) ()

    let q_raise : lambda_stamp -> carries -> result
      = fun stamp x ->
      Callcc.callcc (fun (k : gives Callcc.cont) -> raise (Q_E (stamp, x, k, [])))
  end
  let reset = Q.reset

   (* A-exceptions: simple restartable exceptions allowing us to change
      our mind about the argument of a function in mid-computation. *)
  module A = struct
    type 'a cont = 'a Callcc.cont
    type carries = int
    type gives = forest
    type 'a handler = carries -> (gives -> 'a) -> 'a
    exception A_E of carries * gives cont

    let a_try : (unit -> 'a) -> 'a handler -> 'a
      = fun code handler ->
      try code () with
      | A_E (x, k) -> handler x (Callcc.throw k)

    let a_raise : carries -> 'b
      = fun x -> Callcc.callcc (fun k -> raise (A_E (x,k)))

  (* A little recursion to smuggle a forest F into the computation of F'.
     Timestamps are used to distinguish between different A_raise:s.

     Known inefficiency: we get nested calls to A_plugin accumulating
     as we move down the forest (this happens even with lambda id).
     Getting rid of these should be possible, but not trivial. *)

    let rec a_plugin : carries -> forest -> forest -> forest
      = fun t f f' ->
      Forest (fun m ->
          a_try
            (fun () ->
              let (i, f'') = run f' m in
              (i, a_plugin t f f''))
            (fun t' c ->
              if t' = t
              then c f
              else c (a_raise t')))

  end

  (* The abstraction operation itself *)
  let lambda : (forest -> forest) -> forest
    = fun phi ->
    let stamp = ref () in
    let q_try = Q.q_try stamp in
    let q_raise = Q.q_raise stamp in
    (* the main recursion *)
    let rec lambda' : int -> (forest -> forest) -> forest
      = fun timestamp phi ->
      Forest (fun m ->
          try
            (match (q_try (fun () ->
                        phi (A.a_try
                               (fun () -> A.a_raise timestamp)
                               (fun t c ->
                                 if t = timestamp
                                 then c (Forest (fun n -> q_raise (n, t)))
                                 else c (A.a_raise t))) % m))
             with
             | (a, f') -> (ans a, lambda' timestamp (fun f -> A.a_plugin timestamp f f')))
          with
          | Q.Q_exn (s, (n, timestamp'), c) ->
             if s <> stamp
             then raise (Q.Q_exn (s, (n, timestamp'), c))
             else let q = quest (pair n (nat timestamp)) in
                  (q, lambda' (timestamp+1)
                        (fun f -> Forest (fun p -> c (p, f)))))
    in lambda' 0 phi
end
