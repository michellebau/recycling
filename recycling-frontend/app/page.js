
import Image from 'next/image'
import styles from './page.module.css'
import RecyclingInstructions from '../components/RecyclingInstructions.client';

export default function Home() {

  return (
    <main className={styles.main}>
      {/* Introducing the new header and description */}
      <header className={styles.header}>
          <h1>Welcome to Recycling Assistant</h1>
          <p>This web app helps identify waste items from your uploaded images and provides recycling instructions. To get started, simply upload a picture of the waste item, and we'll guide you through its recycling process.</p>
      </header>

      <div>
      <RecyclingInstructions />
      </div>
      <div className={styles.grid}>
        <a
          href="https://vercel.com?utm_source=create-next-app&utm_medium=appdir-template&utm_campaign=create-next-app"
          className={styles.card}
          target="_blank"
          rel="noopener noreferrer"
        >
          <h2>
            Vercel <span>-&gt;</span>
          </h2>
          <p>Made with Next.js and Vercel.</p>
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
