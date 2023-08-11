'use client'

import { useState } from 'react';
import styles from './RecyclingInstructions.module.css';


function RecyclingInstructions() {
    const [steps, setSteps] = useState(null);
    const [isLoading, setIsLoading] = useState(false);

    const onSubmit = async (event) => {
        event.preventDefault();

        const formData = new FormData();
        formData.append('file', event.target.file.files[0]);

        setIsLoading(true);
        try {
            const response = await fetch('http://localhost:5000/identify_waste', {
                method: 'POST',
                body: formData
            });            

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const data = await response.json();
            console.log("API Response:", data);
            setSteps(data.steps);
        } catch (error) {
            console.error("There was a problem with the fetch operation:", error.message);
        } finally {
            setIsLoading(false);
        }
    }

    return (
        <div>
            <form onSubmit={onSubmit} className={styles.recyclingForm}>
                <label className={styles.labelStyle}>
                    Upload an Image:
                    <input type="file" name="file" accept="image/*" />
                </label>
                <button type="submit" className={styles.buttonStyle}>Submit</button>
            </form>

            {isLoading && <p>Loading...</p>}
            {steps && <div>
    <h2>Recycling Instructions:</h2>
    <ul>
        {steps.map((step, index) => <li key={index}>{step}</li>)}
    </ul>
</div>}

        </div>
    );
}

export default RecyclingInstructions;
