////////////////////////////////////////////////////////
// from Javi's library "common.cc"
// RANDOM NUMBERS       
// We'll use Mersenne Twister 19937 (See GSL-documentation, code is from there)
#define MTRNG_N 624
#define MTRNG_M 397
#define MTRNG_RAND_MAX 0xffffffffUL
static const unsigned long MTRNG_UPPER_MASK = 0x80000000UL;
static const unsigned long MTRNG_LOWER_MASK = 0x7fffffffUL;

typedef struct
{
     unsigned long mt[MTRNG_N];
     int mti;
}mtrng_state_t;

mtrng_state_t *mtrng_state;

void open_rng(unsigned long int s)
{
     mtrng_state=(mtrng_state_t*)malloc(sizeof(mtrng_state_t));
     if (s == 0)
	  s = 4357;   /* the default seed is 4357 */
     mtrng_state->mt[0]= s & 0xffffffffUL;
     int i;
     for (i=1; i<MTRNG_N; i++)
     {
	 /* See Knuth's "Art of Computer Programming" Vol. 2, 3rd
	    Ed. p.106 for multiplier. */
	  mtrng_state->mt[i] =
	       (1812433253UL * 
		(mtrng_state->mt[i-1] ^ (mtrng_state->mt[i-1] >> 30)) + i);
	  mtrng_state->mt[i] &= 0xffffffffUL;
     }
     mtrng_state->mti = i;
}

void close_rng()
{
     free(mtrng_state);
}

unsigned long int get_deviate()
{
     unsigned long k;
     unsigned long int *const mt = mtrng_state->mt;
#define MTRNG_MAGIC(y) (((y)&0x1) ? 0x9908b0dfUL : 0)
     if (mtrng_state->mti >= MTRNG_N)
     {   // generate N words at one time 
	  int kk;
	  for (kk=0; kk<MTRNG_N-MTRNG_M; kk++)
	  {
	       unsigned long y = (mt[kk] & MTRNG_UPPER_MASK) | 
		    (mt[kk + 1] & MTRNG_LOWER_MASK);
	       mt[kk] = mt[kk + MTRNG_M] ^ (y >> 1) ^ MTRNG_MAGIC(y);
	  }
	  for (; kk<MTRNG_N-1; kk++)
	  {
	       unsigned long y = (mt[kk] & MTRNG_UPPER_MASK) | 
		    (mt[kk + 1] & MTRNG_LOWER_MASK);
	       mt[kk] = mt[kk + (MTRNG_M - MTRNG_N)] ^ (y >> 1) ^ 
		    MTRNG_MAGIC(y);
	  }
	  {
	       unsigned long y = (mt[MTRNG_N - 1] & MTRNG_UPPER_MASK) | 
		    (mt[0] & MTRNG_LOWER_MASK);
	       mt[MTRNG_N - 1] = mt[MTRNG_M - 1] ^ (y >> 1) ^ MTRNG_MAGIC(y);
	  }
	  mtrng_state->mti = 0;
     }
     // Tempering
     k = mt[mtrng_state->mti];
     k ^= (k >> 11);
     k ^= (k << 7) & 0x9d2c5680UL;
     k ^= (k << 15) & 0xefc60000UL;
     k ^= (k >> 18);
     mtrng_state->mti++;
     return k;
}

double rand_double()
{
     return get_deviate() / 4294967296.0 ;
}
