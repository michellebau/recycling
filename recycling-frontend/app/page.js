
import Image from 'next/image'
import styles from './page.module.css'
import RecyclingInstructions from '../components/RecyclingInstructions.client';

export default function Home() {

  return (
    <main className={styles.main}>
      <div className={styles.description}>
        <div>
          <a
            href="https://vercel.com?utm_source=create-next-app&utm_medium=appdir-template&utm_campaign=create-next-app"
            target="_blank"
            rel="noopener noreferrer"
          >
            By{' '}
            <Image
              src="/vercel.svg"
              alt="Vercel Logo"
              className={styles.vercelLogo}
              width={100}
              height={24}
              priority
            />
          </a>
        </div>
      </div>

      <div>
      <RecyclingInstructions />
      {/* Other components or content */}
      </div>
      <div className={styles.grid}>
        <a
          href=""
          className={styles.card}
          target="_blank"
          rel="noopener noreferrer"
        >
          <h2>
            Docs <span>-&gt;</span>
          </h2>
          <p>Find in-depth information about Next.js features and API.</p>
        </a>

        <a
          href="https://www.call2recycle.org/locator/"
          className={styles.card}
          target="_blank"
          rel="noopener noreferrer"
        >
          <h2>
            Call2Recycle <span>-&gt;</span>
          </h2>
          <p>Navigate to Call2Recycle, which is the country&#39;s leading consumer battery recycling and stewardship program.</p>
        </a>

        <a
          href="https://search.earth911.com"
          className={styles.card}
          target="_blank"
          rel="noopener noreferrer"
        >
          <h2>
            Earth911 <span>-&gt;</span>
          </h2>
          <p>Utilize Earth 911&#39;s Recycling Search. Earth911 maintains one of North America&#39;s most extensive recycling databases</p>
        </a>

        <a
          href="https://www.epa.gov/recycle/how-do-i-recycle-common-recyclables"
          className={styles.card}
          target="_blank"
          rel="noopener noreferrer"
        >
          <h2>
            EPA <span>-&gt;</span>
          </h2>
          <p>
            Learn more about recycling from the U.S. Environmental Protection Agency.
          </p>
        </a>
      </div>
    </main>
  )
}
